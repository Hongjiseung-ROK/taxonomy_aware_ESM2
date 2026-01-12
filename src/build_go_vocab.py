import json
import os
from pathlib import Path
from tqdm import tqdm

from goatools.obo_parser import GODag

def parse_obo_terms(obo_path):
    """
    Parses OBO file using goatools and returns a set of valid (non-obsolete) GO ids.
    """
    print(f"Loading OBO file via goatools: {obo_path}")
    godag = GODag(obo_path)
    
    valid_terms = set()
    for term in godag.values():
        if not term.is_obsolete:
            valid_terms.add(term.item_id)
            
    return valid_terms

def build_vocab(obo_path, output_path):
    if not os.path.exists(obo_path):
        print(f"Error: OBO file not found at {obo_path}")
        return

    valid_terms = parse_obo_terms(obo_path)
    print(f"Total valid (non-obsolete) GO terms: {len(valid_terms)}")
    
    # Sort for determinism
    sorted_terms = sorted(list(valid_terms))
    
    # Map to indices
    term_to_idx = {term: i for i, term in enumerate(sorted_terms)}
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(term_to_idx, f, indent=2)
        
    print(f"Saved vocabulary to {output_path}")

if __name__ == "__main__":
    # Path to OBO file
    obo_path = Path("go-basic.obo")
    if not obo_path.exists():
        # Fallback to check if it's in go/ (local dev)
        obo_path = Path("go/go-basic.obo")
    
    # Output to src/go_terms.json (uploaded to Azure)
    output_path = Path("src/go_terms.json")
    
    build_vocab(obo_path, output_path)
