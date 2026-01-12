import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json
import os
from Bio import SeqIO
from Bio import SeqIO
from tqdm import tqdm
import scipy.sparse
import pickle

class ProteinTaxonomyDataset(Dataset):
    def __init__(self, fasta_path, term_path, species_vector_path, go_vocab_path, max_len=1024, esm_tokenizer=None, go_matrix_path=None, go_mapping_path=None):
        """
        Args:
            fasta_path: Path to FASTA file.
            term_path: Path to TSV file with GO annotations (EntryID, term).
            species_vector_path: Path to TSV file with species vectors (TaxID, [v1,v2...]).
            go_vocab_path: Path to JSON file with GO term to index mapping.
            max_len: Max sequence length for tokenizer.
            esm_tokenizer: HuggingFace tokenizer for ESM.
        """
        self.max_len = max_len
        self.tokenizer = esm_tokenizer

        # 1. Load GO Vocab
        print(f"Loading GO vocab from {go_vocab_path}...")
        with open(go_vocab_path, 'r') as f:
            self.go_to_idx = json.load(f)
        self.num_classes = len(self.go_to_idx)

        # 1.2 Load Taxonomy Vocabs (to determine embedding sizes)
        # Expected Ranks for vector: Phylum, Class, Order, Family, Genus, Species, Subspecies
        self.tax_ranks = ["phylum", "class", "order", "family", "genus", "species", "subspecies"]
        self.vocab_sizes = []
        
        # Assume vocab files are in species_vector_path parent dir / "vocab"
        # e.g. .../taxon_embedding/species_vectors.tsv -> .../taxon_embedding/vocab/phylum_vocab.json
        vector_dir = os.path.dirname(species_vector_path)
        vocab_dir = os.path.join(vector_dir, "vocab")
        
        print(f"Loading taxonomy vocabs from {vocab_dir}...")
        for rank in self.tax_ranks:
            v_path = os.path.join(vocab_dir, f"{rank}_vocab.json")
            if os.path.exists(v_path):
                with open(v_path, 'r') as f:
                    v_map = json.load(f)
                    # Size is len(v_map) + padding/unknown handling?
                    # The vectorizer uses the values from these maps.
                    # Max index used is len(v_map) if 1-based and <UNK>=0.
                    # Let's take the max value + 1 to be safe, or len+1.
                    # Usually vocab_map includes <UNK>: 0.
                    # So size is len(v_map).
                    self.vocab_sizes.append(len(v_map) + 1) # Safety buffer +1
                    # print(f"  {rank}: {len(v_map)} terms")
            else:
                print(f"Warning: Vocab file {v_path} not found. Using default 1000.")
                self.vocab_sizes.append(1000)
        
        print(f"Taxonomy Vocab Sizes: {self.vocab_sizes}")
        
        # 1.5 Prepare Propagation Table (if configured)
        self.prop_table = {}
        if go_matrix_path and go_mapping_path and os.path.exists(go_matrix_path) and os.path.exists(go_mapping_path):
            print(f"Enabling GO Term Propagation using {go_matrix_path}...")
            
            # Load mapping
            with open(go_mapping_path, 'rb') as f:
                mappings = pickle.load(f)
            
            print(f"Loaded mappings type: {type(mappings)}")
            
            term_to_matrix_idx = None
            idx_to_term_matrix = None
            
            # Helper to identify dicts
            def is_term_to_idx(d):
                if not isinstance(d, dict) or not d: return False
                k = next(iter(d))
                return isinstance(k, str) and isinstance(d[k], int)
                
            def is_idx_to_term(d):
                if not isinstance(d, dict) or not d: return False
                k = next(iter(d))
                return isinstance(k, int) and isinstance(d[k], str)

            # Search strategy
            candidates = []
            if isinstance(mappings, dict):
                if 'term_to_idx' in mappings: candidates.append(mappings['term_to_idx'])
                if 'idx_to_term' in mappings: candidates.append(mappings['idx_to_term'])
                candidates.append(mappings) 
            elif isinstance(mappings, list):
                if len(mappings) > 0 and isinstance(mappings[0], str):
                    # It's likely just [GO:001, GO:002...] i.e. idx_to_term list
                    print("Found list of strings. Assuming it is idx_to_term.")
                    idx_to_term_matrix = {i: t for i, t in enumerate(mappings)}
                    term_to_matrix_idx = {t: i for i, t in enumerate(mappings)}
                else:
                    # Maybe it's [term_to_idx, idx_to_term] tuple/list?
                    print(f"Mappings is list of length {len(mappings)}")
                    for item in mappings:
                        candidates.append(item)
            
            # If we reconstructed them above, candidates loop might be skipped or redundant matches
            # But let's run candidates check if we haven't found them yet
            
            if term_to_matrix_idx is None:
                for c in candidates:
                    if term_to_matrix_idx is None and is_term_to_idx(c):
                        term_to_matrix_idx = c
                        print(f"Found term_to_idx (size {len(c)})")
                    if idx_to_term_matrix is None and is_idx_to_term(c):
                        idx_to_term_matrix = c
                        print(f"Found idx_to_term (size {len(c)})")
            
            if term_to_matrix_idx is None:
                raise ValueError(f"Could not find term_to_idx (str->int) mapping. Mappings type: {type(mappings)}, Length/Size: {len(mappings) if hasattr(mappings, '__len__') else 'N/A'}")
            if idx_to_term_matrix is None:
                # If missing, we can try to invert term_to_idx
                print("Warning: idx_to_term not found, inferring from term_to_idx.")
                idx_to_term_matrix = {v: k for k, v in term_to_matrix_idx.items()}
            
            # Load matrix (CSR: Rows=Child, Cols=Ancestor)
            # mat[i, j] = 1 if j is ancestor of i
            ancestor_matrix = scipy.sparse.load_npz(go_matrix_path)
            
            # Precompute prop_table: vocab_idx -> set of ancestor vocab_indices
            print("Precomputing propagation map for current vocabulary...")
            count_propagated = 0
            
            for go_term, vocab_idx in tqdm(self.go_to_idx.items(), desc="Prop Mapping"):
                # Default: include self (already represented in matrix, but let's be robust)
                ancestors_vocab_indices = {vocab_idx}
                
                if go_term in term_to_matrix_idx:
                    matrix_idx = term_to_matrix_idx[go_term]
                    
                    # Get ancestors from row
                    # CSR is efficient for row slicing
                    # row = ancestor_matrix.getrow(matrix_idx)
                    # indices = row.indices
                    # Faster direct access if matrix format allows
                    
                    # Slice row
                    start = ancestor_matrix.indptr[matrix_idx]
                    end = ancestor_matrix.indptr[matrix_idx+1]
                    ancestor_matrix_indices = ancestor_matrix.indices[start:end]
                    
                    for anc_mat_idx in ancestor_matrix_indices:
                        anc_term = idx_to_term_matrix[anc_mat_idx]
                        if anc_term in self.go_to_idx:
                            ancestors_vocab_indices.add(self.go_to_idx[anc_term])
                            
                self.prop_table[vocab_idx] = list(ancestors_vocab_indices)
                if len(ancestors_vocab_indices) > 1:
                    count_propagated += 1
                    
            print(f"Propagation map built. {count_propagated}/{self.num_classes} terms have ancestors in vocab.")
        else:
            print("Skipping GO Term Propagation (files not provided or found).")

        # 2. Load Species Vectors (Look up table)
        # Expected format: TaxID \t [1, 5, 20...]
        # We need to parse the list string.
        print(f"Loading species vectors from {species_vector_path}...")
        self.tax_vectors = {}
        with open(species_vector_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    tax_id = int(parts[0])
                    # Parse "[1, 2, 3]" -> [1, 2, 3]
                    vector_str = parts[1]
                    # Simple parsing assuming format is clean
                    vector = json.loads(vector_str) 
                    self.tax_vectors[tax_id] = vector
        
        # 3. Load Annotations
        print(f"Loading annotations from {term_path}...")
        self.annotations = {} # EntryID -> set of GO indices
        
        # Read TSV using pandas for speed
        df = pd.read_csv(term_path, sep='\t')
        
        # Filter terms to only those in our vocab
        # (vocab might be built from train+val, so this check is mostly for safety)
        df = df[df['term'].isin(self.go_to_idx.keys())]
        
        # Group by EntryID
        grouped = df.groupby('EntryID')['term'].apply(list)
        
        for entry_id, terms in grouped.items():
            indices = [self.go_to_idx[t] for t in terms]
            
            # Apply Propagation
            if self.prop_table:
                expanded_indices = set()
                for idx in indices:
                    # Union of all ancestors
                    if idx in self.prop_table:
                        expanded_indices.update(self.prop_table[idx])
                    else:
                        expanded_indices.add(idx)
                indices = list(expanded_indices)
                
            self.annotations[entry_id] = torch.tensor(indices, dtype=torch.long)

        # 4. Load Sequences and Index
        # 4. Load Sequences and Index
        print(f"Indexing sequences from {fasta_path}...")
        # Struct-of-Arrays for memory efficiency
        self.ids = []
        self.tax_ids = []
        self.seqs = []
        
        # We need to iterate FASTA and only keep entries that have annotations
        # Also parse TaxID from header "OX=..."
        
        # Optimization: Read all at once if memory allows, or just store offsets.
        # Given 120k parsed sequences isn't too huge for 64GB+ RAM, list is fine.
        # If sequence string is heavy, we can store just strings.
        
        valid_count = 0
        missing_tax_count = 0
        missing_anno_count = 0
        
        for record in SeqIO.parse(fasta_path, "fasta"):
            entry_id = self._parse_entry_id(record.id)
            
            if entry_id not in self.annotations:
                missing_anno_count += 1
                continue
                
            # Parse TaxID
            tax_id = self._parse_tax_id(record.description)
            if tax_id is None or tax_id not in self.tax_vectors:
                # Fallback or skip?
                # If we don't have a vector for this species, we should probably skip or use UNK.
                # Assuming UNK vector is [0,0,0,0,0,0,0].
                # Let's try to handle it.
                if tax_id is None:
                     # print(f"Warning: No TaxID for {entry_id}")
                     pass
                missing_tax_count += 1
                # Check implementation plan: "Use O(1) Lookup". 
                # If missing, we can use a zero vector? 
                # Ideally we should filtered unseen species out, but let's use a default UNK vector
                tax_id = -1 # Marker for UNK
            
            self.ids.append(entry_id)
            self.tax_ids.append(tax_id)
            self.seqs.append(str(record.seq))
            valid_count += 1
            
        print(f"Loaded {valid_count} sequences.")
        print(f"Skipped {missing_anno_count} due to missing annotations.")
        print(f"Found {missing_tax_count} sequences with missing/unknown TaxID.")

    def _parse_entry_id(self, header_id):
        # sp|Q69383|REC6_HUMAN -> Q69383
        # Or just use the whole ID if it matches the TSV
        # TSV uses "Q69383" (Uniprot Accession) usually.
        parts = header_id.split('|')
        if len(parts) >= 2:
            return parts[1]
        return header_id

    def _parse_tax_id(self, header_desc):
        """
        Extracts TaxID from FASTA header.
        Supports:
        1. >... OX=9606 ...
        2. >EntryID 9606 ... (Space separated)
        """
        try:
            # 1. Look for OX= format
            if "OX=" in header_desc:
                part = header_desc.split("OX=")[1].split(" ")[0]
                return int(part)
            
            # 2. Look for simple space separation (e.g. >Q15046 9606)
            # header_desc typically contains the whole header after >
            parts = header_desc.split()
            if len(parts) >= 2:
                # Check if second part is a pure integer
                potential_taxid = parts[1]
                if potential_taxid.isdigit():
                    return int(potential_taxid)
                    
            return None
        except Exception:
            return None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        seq_str = self.seqs[idx]
        tax_id = self.tax_ids[idx]
        entry_id = self.ids[idx]
        
        # 1. Tokenize Sequence
        # ESM tokenizer expects a list of tuples or list of strings?
        # Expecting 'sequence' string for generic tokenizer call
        encoded = self.tokenizer(
            seq_str,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        # 2. Get Tax Vector
        if tax_id in self.tax_vectors:
            tax_vector = torch.tensor(self.tax_vectors[tax_id], dtype=torch.long)
        else:
            # Zero vector [0,0,0,0,0,0,0]
            tax_vector = torch.zeros(7, dtype=torch.long)
            
        # 3. Get Label (Multi-hot)
        label_indices = self.annotations[entry_id]
        label_vec = torch.zeros(self.num_classes, dtype=torch.float32)
        label_vec[label_indices] = 1.0
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'tax_vector': tax_vector,
            'labels': label_vec,
            'entry_id': entry_id # Evaluation might need this
        }
