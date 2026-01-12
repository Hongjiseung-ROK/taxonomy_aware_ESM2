import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import sys
import json

# Import model
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import TaxonomyAwareESM

RANK_NAMES = {
    0: "Superkingdom",
    1: "Phylum",
    2: "Class",
    3: "Order",
    4: "Family",
    5: "Genus",
    6: "Species"
}

TARGET_SPECIES = {
    "Homo sapiens": 9606,
    "Pongo abelii": 9601,
    "Hylobates muelleri": 9588
}

def load_species_vectors(vector_path):
    vectors = {}
    print(f"Loading species vectors from {vector_path}...")
    with open(vector_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                tax_id = int(parts[0])
                vec = json.loads(parts[1])
                vectors[tax_id] = vec
    return vectors

from adjustText import adjust_text

def visualize_embeddings(model_path, vector_path, output_dir):
    print("Visualizing Taxonomy Embeddings using PCA...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Model Weights
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Load Species Vectors (to find indices for targets)
    try:
        species_vectors = load_species_vectors(vector_path)
    except FileNotFoundError:
        print(f"Error: Vector file not found at {vector_path}")
        return
    
    # Verify targets exist
    target_indices = {} # Name -> [idx_0, idx_1, ... idx_6]
    for name, tax_id in TARGET_SPECIES.items():
        if tax_id in species_vectors:
            target_indices[name] = species_vectors[tax_id]
        else:
            print(f"Warning: TaxID {tax_id} ({name}) not found in species vectors.")

    # 3. Visualize per Rank
    for rank_idx in range(7):
        key = f"tax_embeddings.{rank_idx}.weight"
        if key not in state_dict:
            print(f"Missing weight for rank {rank_idx}")
            continue
            
        weight = state_dict[key].numpy()
        
        # PCA
        pca = PCA(n_components=2)
        transformed = pca.fit_transform(weight)
        
        # Plot
        plt.figure(figsize=(15, 15)) # Increased size
        
        # Background: All points (light grey)
        plt.scatter(transformed[:, 0], transformed[:, 1], c='lightgrey', alpha=0.3, s=20, label='Others') # lighter alpha, slightly larger s
        
        # Highlight Targets with Manual Positioning
        colors = ['red', 'blue', 'green']
        
        # Manual offsets/alignments for the 3 specific targets in order:
        # 1. Homo sapiens -> Top Right
        # 2. Pongo abelii -> Top Left
        # 3. Hylobates muelleri -> Bottom Right
        manual_positions = [
            {'xytext': (20, 20), 'ha': 'left', 'va': 'bottom'},  # Top-Right
            {'xytext': (-20, 20), 'ha': 'right', 'va': 'bottom'}, # Top-Left
            {'xytext': (20, -20), 'ha': 'left', 'va': 'top'}     # Bottom-Right
        ]

        for i, (name, indices) in enumerate(target_indices.items()):
            # indices is [v0, v1, ... v6]
            vocab_idx = indices[rank_idx]
            
            if vocab_idx < len(transformed):
                x, y = transformed[vocab_idx]
                
                # Marker
                plt.scatter(x, y, c=colors[i % len(colors)], s=300, edgecolor='black', zorder=10, marker='*')
                
                # Annotation (Manual Position)
                pos = manual_positions[i % len(manual_positions)]
                plt.annotate(name, 
                             xy=(x, y), 
                             xytext=pos['xytext'],
                             textcoords='offset points',
                             ha=pos['ha'], 
                             va=pos['va'],
                             fontsize=16, 
                             fontweight='bold', 
                             color='black',
                             arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
            else:
                print(f"Index {vocab_idx} out of bounds for rank {rank_idx}")

        plt.title(f"Rank {rank_idx}: {RANK_NAMES.get(rank_idx, 'Unknown')} Embedding Space", fontsize=20)
        plt.xlabel("PC1", fontsize=14)
        plt.ylabel("PC2", fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.grid(True, alpha=0.3)
        
        # No automatic adjust_text needed
        
        out_file = os.path.join(output_dir, f"rank_{rank_idx}_{RANK_NAMES[rank_idx]}_pca.png")
        plt.savefig(out_file, dpi=300, bbox_inches='tight') # High DPI
        plt.close()
        print(f"Saved plot to {out_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--vector_path", type=str, default="data/species_vectors.tsv")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    
    visualize_embeddings(args.model_path, args.vector_path, args.output_dir)
