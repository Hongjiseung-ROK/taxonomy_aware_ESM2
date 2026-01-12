import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
from transformers import AutoTokenizer

# Add src to sys.path to allow imports from src/model.py and src/dataset.py
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import your modules
from model import TaxonomyAwareESM
from dataset import ProteinTaxonomyDataset

def analyze_attention(checkpoint_path, data_path, target_ids, device='cuda'):
    print(f"=== Loading Checkpoint: {checkpoint_path} ===")
    
    # 1. Load Model Architecture
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint) # Handle full checkpoint vs state dict only
    
    # Infer num_classes from the classifier weights in state_dict
    # Model definition: self.classifier = nn.Linear(..., num_classes)
    # So key is 'classifier.weight'
    if 'classifier.weight' in state_dict:
        num_classes = state_dict['classifier.weight'].shape[0]
    elif 'classifier.3.weight' in state_dict: # Keeping fallback if user had MLP before
        num_classes = state_dict['classifier.3.weight'].shape[0]
    else:
        # Try to guess or fail
        print("Warning: Could not infer num_classes from state_dict keys. Keys:", state_dict.keys())
        raise KeyError("Could not find classifier weights")
        
    print(f"Detected Num Classes: {num_classes}")
    
    model = TaxonomyAwareESM(
        num_classes=num_classes, 
        pretrained_model_name="facebook/esm2_t6_8M_UR50D",
        freeze_backbone=True
    )
    
    model.load_state_dict(state_dict, strict=False) # STRICT=FALSE because we might miss some metadata
    model.to(device)
    model.eval()
    
    # 2. Hook Cross-Attention to capture weights
    attn_weights_storage = {}
    
    def get_attn_weights(name):
        def hook(module, input, output):
            # output of MultiheadAttention is (attn_output, attn_output_weights)
            # attn_output_weights shape: [Batch, Target_Len, Source_Len]
            # Here: [1, Seq_Len, 7] (7 is Taxonomy Ranks)
            # Note: batch_first=True in model definition
            attn_weights_storage[name] = output[1].detach().cpu()
        return hook
        
    model.cross_attention.register_forward_hook(get_attn_weights('cross_attn'))
    

    # 3. Load Samples from Large Learning Superset
    print("Loading dataset...")
    # Setup Paths
    fasta_path = os.path.join(data_path, "learning_superset", "large_learning_superset.fasta")
    term_path = os.path.join(data_path, "learning_superset", "large_learning_superset_term.tsv")
    species_vec = os.path.join(data_path, "taxon_embedding", "species_vectors.tsv")
    go_vocab = os.path.join("src", "go_terms.json") # Relative to root
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    
    # Lightweight dataset init 
    # Note: parsing the huge fasta might take a moment.
    # To be faster, we could scan for the IDs first, but ProteinTaxonomyDataset loads everything.
    # Given the constraints, let's load it.
    dataset = ProteinTaxonomyDataset(
        fasta_path, term_path, species_vec, go_vocab, 
        max_len=512, 
        esm_tokenizer=tokenizer
    )
    
    print(f"Dataset loaded. Total samples: {len(dataset)}")
    
    # Filter for target IDs
    target_indices = []
    # We need to map ID to index.
    # Dataset doesn't expose a quick map, so we iterate.
    # This might be slow for 75MB fasta, but acceptable for a one-off script.
    
    print(f"Searching for target IDs: {target_ids}")
    found_count = 0
    
    # Optimally, we access dataset information if available.
    # Checking dataset implementation... unique_ids list usually exists.
    if hasattr(dataset, 'protein_ids'):
        for idx, pid in enumerate(dataset.protein_ids):
            if pid in target_ids:
                target_indices.append(idx)
                found_count += 1
    else:
        # Fallback: iterate (slower)
        for i in range(len(dataset)):
            if dataset[i]['entry_id'] in target_ids:
                target_indices.append(i)
                found_count += 1
            if found_count >= len(target_ids):
                break
                
    print(f"Found {len(target_indices)} samples matching targets.")

    # Create output directory
    output_dir = os.path.join("outputs", "attention_map")
    os.makedirs(output_dir, exist_ok=True)

    for idx in target_indices:
        sample = dataset[idx]
        
        # Prepare batch
        input_ids = sample['input_ids'].unsqueeze(0).to(device)
        attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
        tax_vector = sample['tax_vector'].unsqueeze(0).to(device)
        prot_id = sample.get('entry_id', 'Unknown')
        
        print(f"Analyzing Protein ID: {prot_id}")
        
        # 4. Forward Pass
        # Clear previous hooks just in case
        attn_weights_storage.clear()
        
        with torch.no_grad():
            _ = model(input_ids, attention_mask, tax_vector)
            
        # 5. Visualize
        if 'cross_attn' not in attn_weights_storage:
            print(f"Error: Hook did not capture attention weights for {prot_id}.")
            continue

        # attn_weights shape: [1, Seq_Len, 7]
        weights = attn_weights_storage['cross_attn'][0] # Remove batch dim -> [Seq_Len, 7]
        
        # Remove padding from visualization
        seq_len = attention_mask.sum().item()
        weights = weights[:seq_len, :] # [Real_Seq_Len, 7]
        
        # Plot
        plt.figure(figsize=(12, 8))
        # Transpose for easier reading: Y-axis = Taxonomy Ranks, X-axis = Sequence Position
        sns.heatmap(weights.T.numpy(), cmap='viridis', 
                    yticklabels=["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"])
        plt.title(f"Cross-Attention Map - Protein {prot_id}")
        plt.xlabel("Sequence Position (Residues)")
        plt.ylabel("Taxonomic Rank")
        
        save_path = os.path.join(output_dir, f"{prot_id}.png")
        plt.savefig(save_path)
        plt.close() # Close plot to save memory
        print(f"Analysis saved to {save_path}")

    print("\n[Interpretation Guide]")
    print("- Uniform Color? -> Model hasn't learned to distinguish ranks yet.")
    print("- Vertical Stripes? -> Specific residues attend to ALL ranks (Structural importance).")
    print("- Horizontal Stripes? -> Some ranks are universally more important.")
    print("- Scattered Hotspots? -> IDEAL. Specific residues attend to specific ranks.")

if __name__ == "__main__":
    # Target IDs from earlier investigation of large_learning_superset.fasta
    targets = [
        "P0DPQ6", "A0A0C5B5G6", "P40205", 
        "F5H094", "Q6RFH8", "Q0D2H9",
        "L0R8F8", "P0DMW2", "Q6L8H1", "A0A1B0GTW7"
    ]
    
    analyze_attention(
        checkpoint_path="outputs/best_model_fmax.pth", 
        data_path=".", 
        target_ids=targets,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
