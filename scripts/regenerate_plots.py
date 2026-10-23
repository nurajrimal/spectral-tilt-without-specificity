"""
Regenerate plots from saved .pt vectors without needing the model.
Generates similarity matrices, histograms (if X_en/X_fr saved), and full analysis.

Usage: python scripts/regenerate_plots.py results/Qwen3.5-0.8B
"""
import torch
import sys
import os
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import linear_rep_geometry as lrg

# Semantic grouping for concept label ordering
CONCEPT_GROUPS = {
    'verb_morphology': ['[verb - Ved]', '[verb - Ving]', '[verb - 3pSg]', '[verb - V + er]',
                        '[verb - V + able]', '[verb - V + ment]', '[verb - V + tion]',
                        '[Ving - Ved]', '[Ving - 3pSg]', '[3pSg - Ved]'],
    'adj_morphology':  ['[adj - comparative]', '[adj - superlative]', '[adj - un + adj]', '[adj - adj + ly]'],
    'noun_morphology': ['[noun - plural]', '[pronoun - possessive]'],
    'language':        ['[English - French]', '[French - German]', '[French - Spanish]', '[German - Spanish]'],
    'semantic':        ['[male - female]', '[country - capital]', '[thing - color]', '[thing - part]',
                        '[small - big]', '[lower - upper]', '[frequent - infrequent]'],
}


def sort_concept_names(concept_names):
    """Sort concept names by semantic group."""
    sorted_names = []
    for group_concepts in CONCEPT_GROUPS.values():
        for c in group_concepts:
            if c in concept_names:
                sorted_names.append(c)
    for c in concept_names:
        if c not in sorted_names:
            sorted_names.append(c)
    return sorted_names


def regenerate(result_dir):
    # Discover concepts from saved vector files
    raw_files = glob.glob(os.path.join(result_dir, "v_raw_*.pt"))

    concepts_raw = {}
    concepts_raid = {}
    concepts_naive = {}

    for raw_file in raw_files:
        basename = os.path.basename(raw_file)
        name = basename.replace("v_raw_", "").replace(".pt", "")
        raid_file = os.path.join(result_dir, f"v_raid_{name}.pt")
        naive_file = os.path.join(result_dir, f"v_naive_{name}.pt")

        if not os.path.exists(raid_file):
            print(f"Skipping {name}: no matching raid vector")
            continue

        concepts_raw[name] = torch.load(raw_file, map_location='cpu', weights_only=True)
        concepts_raid[name] = torch.load(raid_file, map_location='cpu', weights_only=True)
        if os.path.exists(naive_file):
            concepts_naive[name] = torch.load(naive_file, map_location='cpu', weights_only=True)

    concept_names = sort_concept_names(list(concepts_raw.keys()))
    print(f"Found {len(concept_names)} concepts (naive: {len(concepts_naive)})")

    # Similarity matrices
    raw_vectors = [concepts_raw[n] for n in concept_names]
    raid_vectors = [concepts_raid[n] for n in concept_names]

    lrg.plot_similarity_matrix(raw_vectors, concept_names,
                               "Concept Similarity Matrix (English/Raw)",
                               os.path.join(result_dir, "sim_matrix_raw.png"))
    lrg.plot_similarity_matrix(raid_vectors, concept_names,
                               "Concept Similarity Matrix (French/Raid)",
                               os.path.join(result_dir, "sim_matrix_raid.png"))
    print("Saved similarity matrices.")

    # Histograms (if activations are saved)
    x_en_path = os.path.join(result_dir, "X_en.pt")
    x_fr_path = os.path.join(result_dir, "X_fr.pt")

    if os.path.exists(x_en_path) and os.path.exists(x_fr_path):
        X_en = torch.load(x_en_path, map_location='cpu', weights_only=True)
        X_fr = torch.load(x_fr_path, map_location='cpu', weights_only=True)

        raw_stack = torch.stack([concepts_raw[n] for n in concept_names])
        raid_stack = torch.stack([concepts_raid[n] for n in concept_names])

        lrg.hist_measurement(X_en, X_fr, raw_stack, concept_names,
                             base="English", target="French",
                             save_path=os.path.join(result_dir, "proj_raw_all.png"))
        lrg.hist_measurement(X_en, X_fr, raid_stack, concept_names,
                             base="English", target="French",
                             save_path=os.path.join(result_dir, "proj_raid_all.png"))
        print("Saved projection histograms.")
    else:
        print("No X_en.pt/X_fr.pt found — skipping histograms.")

    # Run full analysis if analyze_raid is available
    try:
        from analyze_raid import analyze
        analyze(result_dir)
    except ImportError:
        print("analyze_raid.py not importable — run it separately.")

    print("Done!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/regenerate_plots.py <result_dir>")
        sys.exit(1)
    regenerate(sys.argv[1])
