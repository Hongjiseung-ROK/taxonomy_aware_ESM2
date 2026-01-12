import json
import os
import sys
from typing import Dict, List, Set

from Bio import SeqIO
from ete3 import NCBITaxa
from tqdm import tqdm

# --- Configuration ---
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

class TaxonomyBuilder:
    def __init__(self):
        """
        Initialize NCBITaxa. 
        Note: The first run will download/update the taxonomy database (~300MB).
        """
        self.ncbi = NCBITaxa()
        self.vocab_sets: Dict[str, Set[str]] = {rank: set() for rank in TARGET_RANKS}
        self.observed_taxids: Set[int] = set()
        
    def extract_taxid_from_header(self, header: str) -> int:
        """
        Extracts NCBI Taxonomy ID from UniProt-style fasta header.
        Format example: "... OX=9606 ..." -> returns 9606
        """
        import re
        try:
            # Parse 'OX=1234' pattern more robustly using regex
            match = re.search(r"OX=(\d+)", header)
            if match:
                return int(match.group(1))
        except Exception:
            pass
        return None

    def get_lineage_names(self, tax_id: int) -> Dict[str, str]:
        """
        Retrieves the full lineage for a given tax_id and maps it to TARGET_RANKS.
        Missing ranks are handled by propagating the parent rank or marking as <UNK>.
        """
        try:
            # Get full lineage of tax IDs
            lineage_ids = self.ncbi.get_lineage(tax_id)
            # Get rank names (e.g., {9606: 'species', ...})
            ranks = self.ncbi.get_rank(lineage_ids)
            # Get scientific names (e.g., {9606: 'Homo sapiens'})
            names = self.ncbi.get_taxid_translator(lineage_ids)
            
            # Map specific ranks to names
            rank_to_name = {}
            for tid in lineage_ids:
                rank = ranks.get(tid)
                if rank == "strain":
                    rank = "subspecies" # Treat strain as subspecies
                
                if rank in TARGET_RANKS:
                    rank_to_name[rank] = names[tid]
            
            # Fill our 7-level vector, handling missing ranks
            result = {}
            for rank in TARGET_RANKS:
                if rank in rank_to_name:
                    name = rank_to_name[rank]
                    self.vocab_sets[rank].add(name) # Add to vocab
                    result[rank] = name
                else:
                    result[rank] = "<UNK>" # Or handle missing data strategy
            
            return result
            
        except ValueError:
            # TaxID not found in DB
            return {rank: "<UNK>" for rank in TARGET_RANKS}

    def build_from_fasta(self, fasta_paths: List[str]):
        """
        Iterates over FASTA files to collect all unique taxonomy terms.
        """
        print(f"Processing FASTA files: {fasta_paths}")
        
        count = 0
        for path in fasta_paths:
            # Use Biopython SeqIO for accurate FASTA parsing
            for record in tqdm(SeqIO.parse(path, "fasta"), desc=f"Reading {os.path.basename(path)}"):
                tax_id = self.extract_taxid_from_header(record.description)
                if tax_id:
                    self.observed_taxids.add(tax_id)
                    self.get_lineage_names(tax_id) # This updates self.vocab_sets
                    count += 1
        
        print(f"Processed {count} sequences from FASTA.")

    def build_from_tsv(self, tsv_paths: List[str]):
        """
        Iterates over TSV files to collect unique TaxIDs and update vocab.
        Expected format: TaxID \t SpeciesName (Header skipped if present)
        """
        if not tsv_paths:
            return

        print(f"Processing TSV files: {tsv_paths}")
        count = 0
        for path in tsv_paths:
            with open(path, 'r') as f:
                # Basic check for header
                first_line = f.readline()
                # If first line looks like a header (not an int), skip it. 
                # Otherwise reset seeking.
                try:
                    int(first_line.split('\t')[0])
                    f.seek(0) # It's data
                except ValueError:
                    pass # It's a header, keep reading from next line

                for line in f:
                    parts = line.strip().split('\t')
                    if parts:
                        try:
                            tax_id = int(parts[0])
                            self.observed_taxids.add(tax_id)
                            self.get_lineage_names(tax_id)
                            count += 1
                        except ValueError:
                            pass
        print(f"Processed {count} additional IDs from TSV.")

    def save_vocab(self, output_dir: str):
        """
        Saves the vocabulary dictionaries to JSON files.
        0 index is reserved for <UNK>.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save Observed TaxIDs
        observed_path = os.path.join(output_dir, "observed_taxids.json")
        with open(observed_path, "w") as f:
            json.dump(list(self.observed_taxids), f)
        print(f"Saved {len(self.observed_taxids)} observed TaxIDs to {observed_path}")
        
        for rank in TARGET_RANKS:
            # Sort for determinism
            unique_terms = sorted(list(self.vocab_sets[rank]))
            
            # Create mapping: term -> id
            # Start ID from 1, reserve 0 for padding/unknown
            vocab_map = {"<UNK>": 0}
            for idx, term in enumerate(unique_terms, 1):
                vocab_map[term] = idx
            
            # Save to JSON
            filename = os.path.join(output_dir, f"{rank}_vocab.json")
            with open(filename, "w") as f:
                json.dump(vocab_map, f, indent=2)
            
            print(f"Saved {rank} vocab with {len(vocab_map)} terms to {filename}")

# --- Execution ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build taxonomy vocabulary from FASTA files.")
    parser.add_argument("--input_fasta", nargs='+', required=True, help="Path to input FASTA file(s)")
    parser.add_argument("--input_tsv", nargs='+', help="Path to input TSV file(s) with TaxIDs")
    parser.add_argument("--output_dir", required=True, help="Directory to save vocabulary JSON files")
    
    args = parser.parse_args()
    
    # Validate input files
    for f in args.input_fasta:
        if not os.path.exists(f):
            print(f"Error: Input file at {f} not found.")
            sys.exit(1)
    
    if args.input_tsv:
        for f in args.input_tsv:
            if not os.path.exists(f):
                print(f"Error: Input file at {f} not found.")
                sys.exit(1)

    builder = TaxonomyBuilder()
    builder.build_from_fasta(args.input_fasta)
    if args.input_tsv:
        builder.build_from_tsv(args.input_tsv)
        
    builder.save_vocab(args.output_dir)
