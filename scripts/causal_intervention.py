#!/usr/bin/env python3
"""
Causal Intervention Experiment — The actual "Raid" test.

For each model, for select concepts:
  1. Load model and concept vectors (raw, WCA-transported, naive-transported)
  2. Generate text with steering: h ← h + α·v at a target layer
  3. Measure if generation shifts in expected direction
  4. Compare WCA vs Naive transported vectors

Metrics:
  - Perplexity shift (does the vector degrade fluency?)
  - Concept probe accuracy (does the concept actually flip?)
  - Cross-entropy with concept-consistent output

Usage: python scripts/causal_intervention.py --model Qwen/Qwen3-0.6B --pair en-fr
"""
import torch
import numpy as np
import json
import os
import sys
import gc
import glob
import argparse
from collections import defaultdict
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import linear_rep_geometry as lrg

# ──────────────────────────────────────────────────────────
# Test Prompts for Concepts (all 27 concepts)
# ──────────────────────────────────────────────────────────
# Each concept has:
#   test_prompts: sentences where steering should shift output
#   pos_keywords: tokens expected when steering toward the SECOND word in the pair
#   neg_keywords: tokens expected when steering toward the FIRST word in the pair
# Convention: concept "[A - B]" means pos_keywords relate to B, neg_keywords to A.

CONCEPT_TESTS = {
    # ── Verb Morphology (10 concepts) ──
    '[verb - Ved]': {
        'test_prompts': [
            "Yesterday the team",
            "Last week she",
            "Before the meeting they",
            "In the past he",
            "Earlier that day we",
        ],
        'pos_keywords': ['accepted', 'achieved', 'created', 'decided', 'discovered',
                         'enjoyed', 'followed', 'improved', 'received', 'asked', 'agreed'],
        'neg_keywords': ['accept', 'achieve', 'create', 'decide', 'discover',
                         'enjoy', 'follow', 'improve', 'receive', 'ask', 'agree'],
    },
    '[verb - Ving]': {
        'test_prompts': [
            "She is currently",
            "They are busy",
            "He keeps on",
            "We have been",
            "The children started",
        ],
        'pos_keywords': ['achieving', 'creating', 'developing', 'following', 'improving',
                         'learning', 'providing', 'running', 'working', 'including'],
        'neg_keywords': ['achieve', 'create', 'develop', 'follow', 'improve',
                         'learn', 'provide', 'run', 'work', 'include'],
    },
    '[verb - 3pSg]': {
        'test_prompts': [
            "He usually",
            "She always",
            "The system",
            "Every morning he",
            "The professor",
        ],
        'pos_keywords': ['accepts', 'achieves', 'allows', 'appears', 'creates',
                         'includes', 'provides', 'requires', 'seems', 'tells'],
        'neg_keywords': ['accept', 'achieve', 'allow', 'appear', 'create',
                         'include', 'provide', 'require', 'seem', 'tell'],
    },
    '[verb - V + er]': {
        'test_prompts': [
            "The person who performs is called a",
            "Someone who teaches is a",
            "A person who writes is a",
            "One who manages others is a",
            "The one who speaks is the",
        ],
        'pos_keywords': ['teacher', 'writer', 'speaker', 'manager', 'performer',
                         'listener', 'explorer', 'leader', 'developer', 'publisher'],
        'neg_keywords': ['teach', 'write', 'speak', 'manage', 'perform',
                         'listen', 'explore', 'lead', 'develop', 'publish'],
    },
    '[verb - V + able]': {
        'test_prompts': [
            "This task is completely",
            "The results are",
            "The problem is quite",
            "This product is very",
            "The plan was deemed",
        ],
        'pos_keywords': ['acceptable', 'achievable', 'affordable', 'believable', 'enjoyable',
                         'manageable', 'predictable', 'sustainable', 'understandable', 'reliable'],
        'neg_keywords': ['accept', 'achieve', 'afford', 'believe', 'enjoy',
                         'manage', 'predict', 'sustain', 'understand', 'rely'],
    },
    '[verb - V + ment]': {
        'test_prompts': [
            "The company announced a new",
            "This represents a major",
            "They celebrated the",
            "The government issued a formal",
            "The board approved the",
        ],
        'pos_keywords': ['achievement', 'agreement', 'announcement', 'development',
                         'improvement', 'investment', 'management', 'requirement',
                         'establishment', 'commitment'],
        'neg_keywords': ['achieve', 'agree', 'announce', 'develop', 'improve',
                         'invest', 'manage', 'require', 'establish', 'commit'],
    },
    '[verb - V + tion]': {
        'test_prompts': [
            "The official document required",
            "This process involves",
            "The committee reviewed the",
            "We need a thorough",
            "The research led to a new",
        ],
        'pos_keywords': ['examination', 'exploration', 'imagination', 'installation',
                         'observation', 'organization', 'preparation', 'realization',
                         'determination', 'declaration'],
        'neg_keywords': ['examine', 'explore', 'imagine', 'install', 'observe',
                         'organize', 'prepare', 'realize', 'determine', 'declare'],
    },
    '[Ving - Ved]': {
        'test_prompts': [
            "The project was",
            "After the process the result was",
            "The report stated it was",
            "By the end it had been",
            "The task was successfully",
        ],
        'pos_keywords': ['added', 'agreed', 'created', 'decided', 'developed',
                         'established', 'included', 'received', 'sent', 'told'],
        'neg_keywords': ['adding', 'agreeing', 'creating', 'deciding', 'developing',
                         'establishing', 'including', 'receiving', 'sending', 'telling'],
    },
    '[Ving - 3pSg]': {
        'test_prompts': [
            "He typically",
            "The system regularly",
            "She often",
            "The machine",
            "This approach",
        ],
        'pos_keywords': ['adds', 'allows', 'appears', 'includes', 'involves',
                         'provides', 'receives', 'remains', 'requires', 'seems'],
        'neg_keywords': ['adding', 'allowing', 'appearing', 'including', 'involving',
                         'providing', 'receiving', 'remaining', 'requiring', 'seeming'],
    },
    '[3pSg - Ved]': {
        'test_prompts': [
            "Last year the company",
            "Previously the team",
            "Before that it",
            "In the old days the leader",
            "Once the system",
        ],
        'pos_keywords': ['added', 'agreed', 'allowed', 'appeared', 'created',
                         'decided', 'developed', 'included', 'received', 'told'],
        'neg_keywords': ['adds', 'agrees', 'allows', 'appears', 'creates',
                         'decides', 'develops', 'includes', 'receives', 'tells'],
    },

    # ── Adjective Morphology (4 concepts) ──
    '[adj - comparative]': {
        'test_prompts': [
            "This one is much",
            "Compared to the other it is",
            "The second option is even",
            "After the upgrade it became",
            "Than the previous version it is",
        ],
        'pos_keywords': ['bigger', 'better', 'faster', 'stronger', 'higher',
                         'cheaper', 'easier', 'longer', 'smaller', 'harder'],
        'neg_keywords': ['big', 'good', 'fast', 'strong', 'high',
                         'cheap', 'easy', 'long', 'small', 'hard'],
    },
    '[adj - superlative]': {
        'test_prompts': [
            "This is the most",
            "It was the",
            "Of all options this is the",
            "Among all candidates this is the",
            "In the entire group the",
        ],
        'pos_keywords': ['biggest', 'best', 'fastest', 'strongest', 'highest',
                         'cheapest', 'easiest', 'longest', 'smallest', 'hardest'],
        'neg_keywords': ['big', 'good', 'fast', 'strong', 'high',
                         'cheap', 'easy', 'long', 'small', 'hard'],
    },
    '[adj - un + adj]': {
        'test_prompts': [
            "The situation was completely",
            "This outcome was",
            "The result was rather",
            "The response seemed",
            "Many found it",
        ],
        'pos_keywords': ['unable', 'unacceptable', 'unavailable', 'unaware', 'uncertain',
                         'uncomfortable', 'unexpected', 'unfortunate', 'unhappy', 'unknown',
                         'unpleasant', 'unpopular', 'unreasonable', 'unusual'],
        'neg_keywords': ['able', 'acceptable', 'available', 'aware', 'certain',
                         'comfortable', 'expected', 'fortunate', 'happy', 'known',
                         'pleasant', 'popular', 'reasonable', 'usual'],
    },
    '[adj - adj + ly]': {
        'test_prompts': [
            "The team performed",
            "She spoke about it",
            "The results improved",
            "He handled the situation",
            "The policy was",
        ],
        'pos_keywords': ['actually', 'apparently', 'critically', 'effectively', 'globally',
                         'immediately', 'obviously', 'previously', 'seriously', 'significantly',
                         'similarly', 'strongly', 'successfully', 'typically'],
        'neg_keywords': ['actual', 'apparent', 'critical', 'effective', 'global',
                         'immediate', 'obvious', 'previous', 'serious', 'significant',
                         'similar', 'strong', 'successful', 'typical'],
    },

    # ── Noun Morphology (2 concepts) ──
    '[noun - plural]': {
        'test_prompts': [
            "I saw one",
            "There was a single",
            "He bought a",
            "She found the",
            "They noticed a",
        ],
        'pos_keywords': ['cars', 'days', 'years', 'ideas', 'problems', 'students',
                         'things', 'people', 'children', 'many', 'several',
                         'items', 'groups', 'systems', 'members'],
        'neg_keywords': ['car', 'day', 'year', 'idea', 'problem', 'student',
                         'thing', 'person', 'child', 'single', 'one', 'a '],
    },
    '[pronoun - possessive]': {
        'test_prompts': [
            "The student forgot to bring",
            "Everyone should do",
            "The team completed",
            "Each person submitted",
            "The traveler packed",
        ],
        'pos_keywords': ['his', 'their', 'our', 'your', 'her', 'its', 'my'],
        'neg_keywords': ['he', 'they', 'we', 'you', 'she', 'it', 'I'],
    },

    # ── Language Pairs (4 concepts) ──
    '[English - French]': {
        'test_prompts': [
            "The word for house is",
            "In another language cat is",
            "The translation of water is",
            "The foreign word for book is",
            "In that language dog means",
        ],
        'pos_keywords': ['maison', 'chat', 'eau', 'livre', 'chien', 'pain', 'mère',
                         'voiture', 'le', 'la', 'les', 'des', 'un', 'une', 'du'],
        'neg_keywords': ['house', 'cat', 'water', 'book', 'dog', 'bread', 'mother',
                         'car', 'the', 'a', 'an', 'of'],
    },
    '[French - German]': {
        'test_prompts': [
            "The word for house is",
            "In another language cat is",
            "The translation of water is",
            "The foreign word for book is",
            "In that language dog means",
        ],
        'pos_keywords': ['Haus', 'Katze', 'Wasser', 'Buch', 'Hund', 'Brot', 'Mutter',
                         'Auto', 'der', 'die', 'das', 'ein', 'eine', 'und', 'ist'],
        'neg_keywords': ['maison', 'chat', 'eau', 'livre', 'chien', 'pain', 'mère',
                         'voiture', 'le', 'la', 'les', 'des', 'un', 'une'],
    },
    '[French - Spanish]': {
        'test_prompts': [
            "The word for house is",
            "In another language cat is",
            "The translation of water is",
            "The foreign word for book is",
            "In that language dog means",
        ],
        'pos_keywords': ['casa', 'gato', 'agua', 'libro', 'perro', 'pan', 'madre',
                         'coche', 'el', 'la', 'los', 'las', 'un', 'una', 'del', 'es'],
        'neg_keywords': ['maison', 'chat', 'eau', 'livre', 'chien', 'pain', 'mère',
                         'voiture', 'le', 'la', 'les', 'des'],
    },
    '[German - Spanish]': {
        'test_prompts': [
            "The word for house is",
            "In another language cat is",
            "The translation of water is",
            "The foreign word for book is",
            "In that language dog means",
        ],
        'pos_keywords': ['casa', 'gato', 'agua', 'libro', 'perro', 'pan', 'madre',
                         'coche', 'el', 'la', 'los', 'las', 'un', 'una', 'es'],
        'neg_keywords': ['Haus', 'Katze', 'Wasser', 'Buch', 'Hund', 'Brot', 'Mutter',
                         'Auto', 'der', 'die', 'das', 'ein', 'eine'],
    },

    # ── Semantic (7 concepts) ──
    '[male - female]': {
        'test_prompts': [
            "The doctor walked into the room and",
            "The teacher told the students that",
            "The CEO announced that",
            "The nurse carefully checked the",
            "My friend said that",
        ],
        'pos_keywords': ['he', 'his', 'him', 'man', 'boy', 'father', 'sir', 'mr', 'himself'],
        'neg_keywords': ['she', 'her', 'woman', 'girl', 'mother', 'madam', 'mrs', 'ms', 'herself'],
    },
    '[country - capital]': {
        'test_prompts': [
            "The most famous city in France is",
            "People travel to",
            "The government headquarters is located in",
            "The capital of Germany is",
            "Tourists often visit",
        ],
        'pos_keywords': ['paris', 'london', 'berlin', 'rome', 'madrid', 'capital', 'city',
                         'tokyo', 'washington', 'moscow', 'beijing'],
        'neg_keywords': ['france', 'germany', 'italy', 'spain', 'england', 'country', 'nation',
                         'japan', 'america', 'russia', 'china'],
    },
    '[thing - color]': {
        'test_prompts': [
            "The apple is typically",
            "The sky appears",
            "Snow is usually",
            "Grass is naturally",
            "Coal is",
        ],
        'pos_keywords': ['red', 'blue', 'green', 'white', 'black', 'yellow', 'brown',
                         'orange', 'pink', 'gray', 'grey', 'purple'],
        'neg_keywords': ['apple', 'sky', 'snow', 'grass', 'coal', 'tomato', 'ocean',
                         'milk', 'cherry', 'cloud', 'emerald', 'sugar'],
    },
    '[thing - part]': {
        'test_prompts': [
            "The most important part of a car is the",
            "A bird is covered in",
            "The main component of a guitar is the",
            "A flower has a delicate",
            "A door swings on its",
        ],
        'pos_keywords': ['engine', 'feathers', 'string', 'petal', 'hinge', 'trigger',
                         'button', 'blade', 'seat', 'keyboard', 'wheel', 'sleeve'],
        'neg_keywords': ['car', 'bird', 'guitar', 'flower', 'door', 'gun',
                         'shirt', 'sword', 'chair', 'piano', 'bus', 'dress'],
    },
    '[small - big]': {
        'test_prompts': [
            "The animal was very",
            "I picked up the",
            "The size of the building was",
            "She described the room as",
            "The object appeared to be",
        ],
        'pos_keywords': ['tiny', 'small', 'little', 'miniature', 'compact', 'minor', 'slight'],
        'neg_keywords': ['large', 'big', 'huge', 'enormous', 'massive', 'giant', 'great', 'vast'],
    },
    '[lower - upper]': {
        'test_prompts': [
            "The letter was written in",
            "Please type your name in",
            "The title appeared in",
            "The word was displayed in",
            "The sign read in",
        ],
        'pos_keywords': ['uppercase', 'upper', 'capital', 'CAPS', 'capitalized', 'majuscule'],
        'neg_keywords': ['lowercase', 'lower', 'small letter', 'minuscule'],
    },
    '[frequent - infrequent]': {
        'test_prompts': [
            "A more formal word for angry is",
            "A sophisticated synonym for big is",
            "The elevated term for happy is",
            "A literary word for cold is",
            "An uncommon way to say fast is",
        ],
        'pos_keywords': ['irate', 'gorgeous', 'ecstatic', 'frigid', 'swift', 'terrible',
                         'purchase', 'consume', 'affluent', 'melancholy', 'perceive',
                         'robust', 'stroll', 'utilize', 'adversary'],
        'neg_keywords': ['angry', 'beautiful', 'happy', 'cold', 'fast', 'bad',
                         'buy', 'eat', 'rich', 'sad', 'see',
                         'strong', 'walk', 'use', 'enemy'],
    },
}


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────
# Steering Hook
# ──────────────────────────────────────────────────────────
class SteeringHook:
    """Adds a steering vector to residual stream at a specific layer."""
    
    def __init__(self, vector, alpha=1.0, position=-1):
        """
        vector: (d_model,) steering direction
        alpha: scaling factor
        position: token position to steer (-1 for last)
        """
        self.vector = vector.detach().float()
        self.alpha = alpha
        self.position = position
        self.handle = None
    
    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        
        pos = self.position if self.position >= 0 else h.shape[1] + self.position
        pos = min(pos, h.shape[1] - 1)
        
        v = self.vector.to(h.device, dtype=h.dtype)
        h[:, pos, :] += self.alpha * v
        
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h
    
    def attach(self, layer_module):
        self.handle = layer_module.register_forward_hook(self.hook_fn)
        return self
    
    def remove(self):
        if self.handle:
            self.handle.remove()


def get_layers(model):
    """Get transformer layers for steering injection."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    return None


def generate_text(model, tokenizer, prompt, max_new_tokens=30, temperature=0.7, top_p=0.9):
    """Generate text from a prompt."""
    inputs = tokenizer(prompt, return_tensors='pt', padding=False)
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    generated = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated.strip()


def compute_logit_shift(model, tokenizer, prompt, concept_vector, target_layer, alpha=1.0):
    """Measure how much a steering vector shifts the next-token distribution."""
    layers = get_layers(model)
    if layers is None or target_layer >= len(layers):
        return {}
    
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors='pt', padding=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Baseline logits
    with torch.no_grad():
        baseline_logits = model(**inputs).logits[:, -1, :]
        baseline_probs = torch.softmax(baseline_logits, dim=-1)
    
    # Steered logits
    hook = SteeringHook(concept_vector, alpha=alpha)
    hook.attach(layers[target_layer])
    with torch.no_grad():
        steered_logits = model(**inputs).logits[:, -1, :]
        steered_probs = torch.softmax(steered_logits, dim=-1)
    hook.remove()
    
    # KL divergence
    kl_div = torch.sum(steered_probs * (torch.log(steered_probs.clamp(min=1e-10)) - 
                                         torch.log(baseline_probs.clamp(min=1e-10)))).item()
    
    # Top-k token changes
    _, top_baseline = torch.topk(baseline_probs, 10)
    _, top_steered = torch.topk(steered_probs, 10)
    
    return {
        'kl_divergence': kl_div,
        'top_baseline': [tokenizer.decode([t]) for t in top_baseline[0]],
        'top_steered': [tokenizer.decode([t]) for t in top_steered[0]],
        'baseline_probs': baseline_probs[0].cpu(),
        'steered_probs': steered_probs[0].cpu(),
    }


def keyword_score(text, keywords):
    """Count keyword occurrences (case insensitive)."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


# ──────────────────────────────────────────────────────────
# Main Experiment
# ──────────────────────────────────────────────────────────
def run_intervention(model_path, pair, results_root="results", alphas=[0.5, 1.0, 2.0, 5.0]):
    model_name = model_path.split("/")[-1]
    pair_dir = os.path.join(results_root, model_name, pair)
    
    if not os.path.exists(pair_dir):
        print(f"No results found at {pair_dir}")
        return
    
    output_dir = os.path.join(pair_dir, "intervention")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"  Causal Intervention Experiment")
    print(f"  Model: {model_name}")
    print(f"  Pair: {pair}")
    print(f"{'='*60}")
    
    # Load model
    print("\n[1] Loading model...")
    model, tokenizer, handler = lrg.load_model(model_path)
    device = next(model.parameters()).device
    layers = get_layers(model)
    n_layers = len(layers) if layers else 0
    target_layer = n_layers * 3 // 4  # inject at 75% depth
    print(f"  Layers: {n_layers}, steering at layer {target_layer}")
    
    # Load vectors
    print("\n[2] Loading concept vectors...")
    available_concepts = []
    for concept_name in CONCEPT_TESTS:
        rf = os.path.join(pair_dir, f"v_raw_{concept_name}.pt")
        wf = os.path.join(pair_dir, f"v_raid_{concept_name}.pt")
        nf = os.path.join(pair_dir, f"v_naive_{concept_name}.pt")
        if all(os.path.exists(f) for f in [rf, wf, nf]):
            available_concepts.append(concept_name)
    
    print(f"  Available concepts: {available_concepts}")
    
    if not available_concepts:
        print("  No matching concepts found!")
        return
    
    # Recompute with λ=0.01 as well
    print("\n[3] Recomputing WCA vectors with λ=0.01...")
    cov_src = cov_tgt = X_src = X_tgt = None
    for src_n, tgt_n in [("cov_src.pt", "cov_tgt.pt"), ("cov_en.pt", "cov_fr.pt")]:
        sp = os.path.join(pair_dir, src_n)
        tp = os.path.join(pair_dir, tgt_n)
        if os.path.exists(sp) and os.path.exists(tp):
            cov_src = torch.load(sp, map_location='cpu', weights_only=True).float()
            cov_tgt = torch.load(tp, map_location='cpu', weights_only=True).float()
            break
    
    for src_n, tgt_n in [("X_src.pt", "X_tgt.pt"), ("X_en.pt", "X_fr.pt")]:
        sp = os.path.join(pair_dir, src_n)
        tp = os.path.join(pair_dir, tgt_n)
        if os.path.exists(sp) and os.path.exists(tp):
            X_src = torch.load(sp, map_location='cpu', weights_only=True).float()
            X_tgt = torch.load(tp, map_location='cpu', weights_only=True).float()
            break
    
    recomputed_01 = {}
    if cov_src is not None and X_src is not None:
        psi_src, _ = lrg.get_whitening_transform(cov_src, reg_lambda=0.01)
        psi_tgt, sqrt_tgt = lrg.get_whitening_transform(cov_tgt, reg_lambda=0.01)
        Q = lrg.solve_whitened_procrustes(X_src, X_tgt, psi_src, psi_tgt)
        for cn in available_concepts:
            v_raw = torch.load(os.path.join(pair_dir, f"v_raw_{cn}.pt"), map_location='cpu', weights_only=True).float()
            recomputed_01[cn] = lrg.transport_concept_vector(v_raw, psi_src, sqrt_tgt, Q).cpu()
        print(f"  Recomputed {len(recomputed_01)} vectors with λ=0.01")
    
    # Run interventions
    print(f"\n[4] Running interventions (α ∈ {alphas})...")
    all_results = []
    
    for concept_name in available_concepts:
        test_data = CONCEPT_TESTS[concept_name]
        v_raw = torch.load(os.path.join(pair_dir, f"v_raw_{concept_name}.pt"), map_location='cpu', weights_only=True).float()
        v_wca = torch.load(os.path.join(pair_dir, f"v_raid_{concept_name}.pt"), map_location='cpu', weights_only=True).float()
        v_naive = torch.load(os.path.join(pair_dir, f"v_naive_{concept_name}.pt"), map_location='cpu', weights_only=True).float()
        v_wca_01 = recomputed_01.get(concept_name, v_wca)
        
        print(f"\n  ── {concept_name} ──")
        
        for alpha in alphas:
            for method, vec, label in [
                ('raw', v_raw, 'Raw (source)'),
                ('naive', v_naive, 'Naive Procrustes'),
                ('wca_1e-3', v_wca, 'WCA (λ=1e-3)'),
                ('wca_0.01', v_wca_01, 'WCA (λ=0.01)'),
            ]:
                kl_scores = []
                pos_shifts = []
                neg_shifts = []
                generations = []
                
                for prompt in test_data['test_prompts']:
                    # Generate baseline
                    base_text = generate_text(model, tokenizer, prompt, max_new_tokens=30)
                    
                    # Generate with positive steering
                    hook = SteeringHook(vec, alpha=alpha)
                    hook.attach(layers[target_layer])
                    pos_text = generate_text(model, tokenizer, prompt, max_new_tokens=30)
                    hook.remove()
                    
                    # Generate with negative steering
                    hook = SteeringHook(vec, alpha=-alpha)
                    hook.attach(layers[target_layer])
                    neg_text = generate_text(model, tokenizer, prompt, max_new_tokens=30)
                    hook.remove()
                    
                    # Logit shift
                    shift = compute_logit_shift(model, tokenizer, prompt, vec, target_layer, alpha=alpha)
                    kl_scores.append(shift.get('kl_divergence', 0))
                    
                    # Keyword scoring
                    pos_shift = keyword_score(pos_text, test_data['pos_keywords']) - keyword_score(base_text, test_data['pos_keywords'])
                    neg_shift = keyword_score(neg_text, test_data['neg_keywords']) - keyword_score(base_text, test_data['neg_keywords'])
                    pos_shifts.append(pos_shift)
                    neg_shifts.append(neg_shift)
                    
                    generations.append({
                        'prompt': prompt,
                        'baseline': base_text,
                        'positive': pos_text,
                        'negative': neg_text,
                    })
                
                result = {
                    'concept': concept_name,
                    'method': method,
                    'label': label,
                    'alpha': alpha,
                    'mean_kl': np.mean(kl_scores),
                    'mean_pos_shift': np.mean(pos_shifts),
                    'mean_neg_shift': np.mean(neg_shifts),
                    'directional_score': np.mean(pos_shifts) + np.mean(neg_shifts),  # both should be positive
                    'generations': generations,
                }
                all_results.append(result)
                
                print(f"    {label:<22} α={alpha:<4} KL={result['mean_kl']:.4f}  DirScore={result['directional_score']:+.2f}")
    
    # Save results
    serializable = []
    for r in all_results:
        s = {k: v for k, v in r.items() if k != 'generations'}
        s['generations'] = r['generations']
        serializable.append(s)
    
    with open(os.path.join(output_dir, "intervention_results.json"), 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    
    print(f"\n  Results saved to {output_dir}/intervention_results.json")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY (α=1.0)")
    print(f"{'='*60}")
    print(f"  {'Concept':<25} {'Method':<22} {'KL':>7} {'DirScore':>10}")
    print(f"  {'-'*68}")
    for r in all_results:
        if r['alpha'] == 1.0:
            print(f"  {r['concept']:<25} {r['label']:<22} {r['mean_kl']:>7.4f} {r['directional_score']:>+10.2f}")
    
    cleanup_gpu()
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Causal Intervention Experiment")
    parser.add_argument("--model", required=True, help="Model path (e.g., Qwen/Qwen3-0.6B)")
    parser.add_argument("--pair", default="en-fr", help="Language pair (default: en-fr)")
    parser.add_argument("--results", default="results", help="Results directory")
    parser.add_argument("--alphas", nargs='+', type=float, default=[0.5, 1.0, 2.0, 5.0])
    args = parser.parse_args()
    
    run_intervention(args.model, args.pair, args.results, args.alphas)
