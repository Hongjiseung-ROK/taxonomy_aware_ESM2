import os
import sys
import json
import argparse
from tqdm import tqdm
from ete3 import NCBITaxa

# Target Ranks for our 7-level hierarchy
TARGET_RANKS = [
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
    "subspecies"
]

class SpeciesVectoriser:
    def __init__(self, vocab_dir):
        self.vocab_dir = vocab_dir
        self.vocab_maps = {}
        self.ncbi = NCBITaxa()
        self.load_vocabs()

    def load_vocabs(self):
        """Loads existing vocabulary JSON files."""
        print(f"Loading vocabularies from {self.vocab_dir}...")
        for rank in TARGET_RANKS:
            vocab_path = os.path.join(self.vocab_dir, f"{rank}_vocab.json")
            if not os.path.exists(vocab_path):
                print(f"Error: Vocabulary file {vocab_path} not found.")
                sys.exit(1)
            
            with open(vocab_path, "r") as f:
                self.vocab_maps[rank] = json.load(f)
        print("Vocabularies loaded.")

    def get_lineage_vector(self, tax_id):
        """Retrieves lineage and converts to vector."""
        try:
            lineage_ids = self.ncbi.get_lineage(tax_id)
            ranks = self.ncbi.get_rank(lineage_ids)
            names = self.ncbi.get_taxid_translator(lineage_ids)
            
            rank_to_name = {}
            for tid in lineage_ids:
                rank = ranks.get(tid)
                if rank == "strain":
                    rank = "subspecies"
                
                if rank in TARGET_RANKS:
                    rank_to_name[rank] = names[tid]
            
            vector = []
            for rank in TARGET_RANKS:
                name = rank_to_name.get(rank, "<UNK>")
                # Use vocab mapping, default to 0 (<UNK>)
                # Note: vocab maps string keys to int values
                term_id = self.vocab_maps[rank].get(name, 0)
                vector.append(term_id)
            
            return vector
            
        except ValueError:
            # TaxID not found
            return [0] * len(TARGET_RANKS)
        except Exception as e:
            # Other errors
            return [0] * len(TARGET_RANKS)

    def vectorize_all(self, output_dir):
        """Iterates through all observed TaxIDs (if available) or species vocabulary to generate vectors."""
        observed_ids_path = os.path.join(self.vocab_dir, "observed_taxids.json")
        tax_ids_to_process = []
        
        if os.path.exists(observed_ids_path):
            print(f"Loading observed TaxIDs from {observed_ids_path}...")
            with open(observed_ids_path, "r") as f:
                tax_ids_to_process = json.load(f)
            # Ensure uniqueness just in case
            tax_ids_to_process = sorted(list(set(tax_ids_to_process)))
            print(f"Loaded {len(tax_ids_to_process)} unique observed TaxIDs.")
        else:
            print("observed_taxids.json not found. Falling back to species vocabulary keys.")
            species_vocab = self.vocab_maps["species"]
            # Fallback: map names to IDs. This is less accurate for strains but keeps old behavior.
            name_to_taxid = self.ncbi.get_name_translator(species_vocab.keys())
            for tax_ids in name_to_taxid.values():
                if tax_ids:
                    tax_ids_to_process.append(tax_ids[0])
            tax_ids_to_process = sorted(list(set(tax_ids_to_process)))

        print(f"Vectorizing {len(tax_ids_to_process)} TaxIDs...")
        
        output_path = os.path.join(output_dir, "species_vectors.tsv")
        
        with open(output_path, "w") as f:
            count = 0
            for tax_id in tqdm(tax_ids_to_process):
                 vector = self.get_lineage_vector(tax_id)
                 
                 # Format: ID \t [1, 2, 3, ...]
                 vector_str = "[" + ", ".join(map(str, vector)) + "]"
                 f.write(f"{tax_id}\t{vector_str}\n")
                 count += 1
                 
        print(f"Saved {count} vectors to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Vectorize species using existing vocabularies.")
    parser.add_argument("--vocab_dir", required=True, help="Directory containing _vocab.json files")
    parser.add_argument("--output_dir", required=True, help="Directory to save species_vectors.tsv")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    vectorizer = SpeciesVectoriser(args.vocab_dir)
    vectorizer.vectorize_all(args.output_dir)

if __name__ == "__main__":
    main()
