import argparse
import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Add CAFA evaluator to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'CAFA-evaluator-PK', 'src'))
try:
    from cafaeval.parser import obo_parser, gt_parser, pred_parser
    from cafaeval.evaluation import evaluate_prediction
    HAS_EVAL = True
except ImportError as e:
    print(f"Warning: Could not import cafaeval: {e}")
    HAS_EVAL = False

from dataset import ProteinTaxonomyDataset
from model import TaxonomyAwareESM, AsymmetricLoss
from asymmetric_loss import load_ia_weights
from transformers import AutoTokenizer

def save_checkpoint(model, optimizer, epoch, metrics, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filename)
    print(f"Saved checkpoint to {filename}")

def run_evaluation(model, valid_loader, ontologies, gt, device, out_dir, epoch):
    model.eval()
    all_preds = []
    
    print(f"Generating predictions for validation set (Epoch {epoch})...")
    
    # Needs valid_loader.dataset.go_to_idx to map back to GO IDs
    idx_to_go = {v: k for k, v in valid_loader.dataset.go_to_idx.items()}
    
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Valid Infer"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tax_vector = batch['tax_vector'].to(device)
            entry_ids = batch['entry_id']
            
            logits = model(input_ids, attention_mask, tax_vector)
            probs = torch.sigmoid(logits)
            probs = probs.cpu().numpy()
            
            for i, entry_id in enumerate(entry_ids):
                row_probs = probs[i]
                # Threshold for sparseness
                indices = np.where(row_probs > 0.01)[0]
                for idx in indices:
                    term = idx_to_go[idx]
                    score = float(row_probs[idx])
                    all_preds.append((entry_id, term, score))
                    
    # Format for cafa_eval
    pred_dir = os.path.join(out_dir, "preds_temp")
    os.makedirs(pred_dir, exist_ok=True)
    pred_path = os.path.join(pred_dir, f"epoch_{epoch}.tsv")
    
    with open(pred_path, 'w') as f:
        for p in all_preds:
            f.write(f"{p[0]}\t{p[1]}\t{p[2]:.5f}\n")
            
    print(f"Saved validation predictions to {pred_path}")
    
    if not HAS_EVAL or ontologies is None:
        return 0, {}
        
    print("Running CAFA Evaluation...")
    try:
        # Prediction Parser
        # prop='max' by default in cafa_eval
        prediction = pred_parser(pred_path, ontologies, gt, prop_mode='max', max_terms=None)
        
        if not prediction:
            print("Warning: No predictions parsed.")
            return 0, {}

        # Evaluate
        tau_arr = np.arange(0.01, 1, 0.01)
        df_res = evaluate_prediction(
            prediction, gt, ontologies, tau_arr, 
            gt_exclude=None, normalization='cafa', n_cpu=4
        )
        
        # Extract Weighted F-max
        # df_res contains columns like 'f', 'f_w', etc.
        # Find max f_w across all namespaces? Usually we care about specific ones (BP, MF, CC).
        # But 'weighted f-max' in CAFA usually refers to the specific metric per namespace.
        # We can sum them or pick one? 
        # Usually we want to save the 'best' stats.
        
        # Let's compute best weighted F-max for each namespace and verify
        # df_res index is usually range 0..N, with columns 'ns', 'tau', ...
        
        best_rows = df_res.loc[df_res.groupby('ns')['f_w'].idxmax()]
        print("Best Weighted F-max per namespace:")
        print(best_rows[['ns', 'tau', 'f_w', 'cov_w']])
        
        # We return the average Weighted F-max across namespaces as a scalar metric?
        # Or just return the whole dataframe and let main decide.
        # For checkpointing, let's use the average of f_w.
        avg_wf_max = best_rows['f_w'].mean()
        
        return avg_wf_max, df_res

    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 0, {}

def evaluate_gpu(model, dataloader, ic_weights, device, thresholds=None, pred_output_path=None, metrics_output_path=None):
    """
    Calculates Weighted F-max using GPU streaming to avoid OOM.
    """
    model.eval()
    
    if thresholds is None:
        thresholds = torch.linspace(0, 1, 101, device=device)
    
    # Initialize accumulators for each threshold
    sum_prec = torch.zeros(len(thresholds), device=device)
    sum_rec = torch.zeros(len(thresholds), device=device)
    
    total_samples = 0
    
    total_samples = 0
    
    # Prepare Prediction Output
    f_pred = None
    if pred_output_path:
        os.makedirs(os.path.dirname(pred_output_path), exist_ok=True)
        f_pred = open(pred_output_path, 'w')
        # Need reverse mapping idx -> GO Term
        idx_to_go = {v: k for k, v in dataloader.dataset.go_to_idx.items()}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="GPU Eval"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tax_vector = batch['tax_vector'].to(device)
            labels = batch['labels'].to(device) # (B, NumClasses)
            entry_ids = batch['entry_id']
            
            # Debug/Fix for ID Truncation issue
            # If entry_ids is a single string (e.g. "C0HM65") but batch > 1, iteration yields chars ('C', '0', ... '5')
            # This causes "5" to be written as ID.
            if isinstance(entry_ids, str):
                # This should only happen for Batch Size 1 if standard collate returns string? 
                # Or if something is broken.
                entry_ids = [entry_ids]
            
            # Ensure it is a list/tuple
            if not isinstance(entry_ids, (list, tuple)):
                 # Convert tensor or other to list if needed, though usually tuple/list
                 if isinstance(entry_ids, torch.Tensor):
                     entry_ids = entry_ids.tolist()
                 else:
                     entry_ids = list(entry_ids)
            
            # Check length Consistency
            if len(entry_ids) != input_ids.size(0):
               # If strict mismatch, maybe we have a problem.
               # But let's just proceed with safe iteration
               pass
            
            # 1. Forward
            logits = model(input_ids, attention_mask, tax_vector)
            probs = torch.sigmoid(logits) # (B, NumClasses)
            
            # Save Predictions (Streamed)
            if f_pred:
                probs_cpu = probs.cpu().numpy()
                for i, entry_id in enumerate(entry_ids):
                    # Sparse write: only > 0.01
                    indices = np.where(probs_cpu[i] > 0.01)[0]
                    for idx in indices:
                        term = idx_to_go[idx]
                        score = probs_cpu[i][idx]
                        f_pred.write(f"{entry_id}\t{term}\t{score:.4f}\n")
            
            # 2. Ground Truth IC (for Recall)
            # labels * weights
            # ic_weights: (NumClasses,)
            true_ic = (labels * ic_weights).sum(dim=1) # (B,)
            
            # Avoid div by zero
            true_ic = torch.maximum(true_ic, torch.tensor(1e-9, device=device))
            
            # 3. Iterate Thresholds (Vectorized over batch? RAM concern dependent)
            # Doing it threshold-by-threshold might be slower but safer for memory if we have many thresholds.
            # But we can broadcast: (B, 1, C) >= (1, T, 1) -> (B, T, C)
            # (B, T, C) is 256 * 100 * 40000 * 1 bit? ~120MB. Safe.
            
            probs_unsqueezed = probs.unsqueeze(1) # (B, 1, C)
            thresholds_unsqueezed = thresholds.view(1, -1, 1) # (1, T, 1)
            
            # Pred Binary: (B, T, C)
            pred_binary = (probs_unsqueezed >= thresholds_unsqueezed).float()
            
            # Weighted Intersection (TP): (B, T, C) * (B, 1, C) * (1, 1, C)
            # We can optimize: (pred_binary * labels) -> TP
            labels_unsqueezed = labels.unsqueeze(1) # (B, 1, C)
            
            # Intersection IC: sum((pred & label) * weight, dim=2)
            # (B, T)
            intersection_ic = (pred_binary * labels_unsqueezed * ic_weights.view(1, 1, -1)).sum(dim=2)
            
            # Union (Prediction) IC: sum(pred * weight, dim=2)
            # (B, T)
            pred_ic = (pred_binary * ic_weights.view(1, 1, -1)).sum(dim=2)
            
            # Precision: Int / Pred
            precision = intersection_ic / (pred_ic + 1e-9)
            
            # Recall: Int / True
            recall = intersection_ic / (true_ic.view(-1, 1) + 1e-9)
            
            # Accumulate
            sum_prec += precision.sum(dim=0)
            sum_rec += recall.sum(dim=0)
            
            total_samples += input_ids.size(0)
            
            # Explicit GC to be safe
            del logits, probs, pred_binary, intersection_ic, pred_ic
            
    if f_pred:
        f_pred.close()
        print(f"Saved predictions to {pred_output_path}")
            
    # Compute Averages
    avg_prec = sum_prec / total_samples
    avg_rec = sum_rec / total_samples
    
    # F1
    f1_scores = 2 * avg_prec * avg_rec / (avg_prec + avg_rec + 1e-9)
    
    best_fmax = f1_scores.max().item()
    best_t_idx = f1_scores.argmax().item()
    best_threshold = thresholds[best_t_idx].item()
    
    
    # Save Metrics Detail
    if metrics_output_path:
        metrics_data = {
            'threshold': thresholds.cpu().numpy(),
            'precision': avg_prec.cpu().numpy(),
            'recall': avg_rec.cpu().numpy(),
            'f1': f1_scores.cpu().numpy()
        }
        pd.DataFrame(metrics_data).to_csv(metrics_output_path, sep='\t', index=False)
        print(f"Saved detailed metrics to {metrics_output_path}")

    return best_fmax, best_threshold, f1_scores.cpu().numpy()

def validate_loss(model, valid_loader, criterion, device):
    model.eval()
    total_loss = 0
    steps = 0
    torch.cuda.empty_cache()
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Valid Loss"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tax_vector = batch['tax_vector'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast():
                logits = model(input_ids, attention_mask, tax_vector)
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            steps += 1
            
    return total_loss / steps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to mounted dataset")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--T_0", type=int, default=10, help="CosineAnnealingWarmRestarts T_0")
    parser.add_argument("--T_mult", type=int, default=1, help="CosineAnnealingWarmRestarts T_mult")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--esm_model_name", type=str, default="facebook/esm2_t33_650M_UR50D", help="ESM model name")
    parser.add_argument("--gamma_neg", type=float, default=4, help="Asymmetric Loss gamma_neg")
    parser.add_argument("--gamma_pos", type=float, default=0, help="Asymmetric Loss gamma_pos")
    parser.add_argument("--clip", type=float, default=0.05, help="Asymmetric Loss clip")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for checkpoints and predictions")
    parser.add_argument("--mlflow_dir", type=str, default="mlruns", help="Directory for MLflow logs")
    
    # LoRA Arguments
    parser.add_argument("--use_lora", type=bool, default=True, help="Use LoRA for ESM backbone")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    
    args = parser.parse_args()

    # Paths
    data_path = Path(args.data_path)
    train_fasta = data_path / "learning_superset" / "large_learning_superset.fasta"
    train_term = data_path / "learning_superset" / "large_learning_superset_term.tsv"
    
    val_fasta = data_path / "validation_superset" / "validation_superset.fasta"
    val_term = data_path / "validation_superset" / "validation_superset_term.tsv"
    
    species_vec = data_path / "taxon_embedding" / "species_vectors.tsv"
    
    # GO Vocab is local in src/go_terms.json
    go_vocab_path = "src/go_terms.json"
    if not os.path.exists(go_vocab_path):
        go_vocab_path = "go_terms.json"
        
    # Evaluation files
    obo_path = data_path / "go_info" / "go-basic.obo"
    ia_path = data_path / "IA.tsv"
    
    # Propagation files
    go_matrix_path = data_path / "go_info" / "go_ancestor_matrix.npz"
    go_mapping_path = data_path / "go_info" / "go_term_mappings.pkl"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Tokenizer
    print(f"Loading tokenizer for: {args.esm_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.esm_model_name)

    # Datasets
    print("Initializing Datasets...")
    train_dataset = ProteinTaxonomyDataset(
        train_fasta, train_term, species_vec, go_vocab_path, max_len=1024, esm_tokenizer=tokenizer,
        go_matrix_path=str(go_matrix_path), go_mapping_path=str(go_mapping_path)
    )
    val_dataset = ProteinTaxonomyDataset(
        val_fasta, val_term, species_vec, go_vocab_path, max_len=1024, esm_tokenizer=tokenizer,
        go_matrix_path=str(go_matrix_path), go_mapping_path=str(go_mapping_path)
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=max(1, args.batch_size // 2), 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )

    # Model
    model = TaxonomyAwareESM(
        num_classes=train_dataset.num_classes, 
        pretrained_model_name=args.esm_model_name,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        vocab_sizes=train_dataset.vocab_sizes
    ).to(device)
    
    criterion = AsymmetricLoss(gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos, clip=args.clip).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.min_lr
    )
    
    scaler = GradScaler()
    
    # Pre-Load Ontology and GT for Evaluation
    ontologies = None
    gt = None
    if HAS_EVAL:
        print("Loading Ontology and Ground Truth...")
        # obo_parser(obo_file, valid_rel, ia_file, no_orphans)
        # Using IA file allows Weighted F-max
        ontologies = obo_parser(
            str(obo_path), 
            ("is_a", "part_of"), 
            str(ia_path) if ia_path.exists() else None, 
            True # no_orphans
        )
        gt = gt_parser(str(val_term), ontologies)
        
    # Load IC Weights for GPU Eval
    print("Loading IC Weights for GPU Evaluation...")
    ic_weights = load_ia_weights(
        str(ia_path) if ia_path.exists() else "IA.tsv",
        train_dataset.go_to_idx,
        train_dataset.num_classes
    ).to(device)
    
    # MLflow init
    import mlflow
    import time
    
    # Configure MLflow
    if args.mlflow_dir:
        mlflow_uri = Path(args.mlflow_dir).resolve().as_uri()
        mlflow.set_tracking_uri(mlflow_uri)
        print(f"MLflow tracking URI: {mlflow_uri}")
    
    mlflow.start_run()
    mlflow.log_params(vars(args))

    best_val_loss = float('inf')
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Best model path for validation loss
    best_model_path = output_dir / "best_model_loss.pth"
    best_wf_max = 0.0

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        # Training
        model.train()
        total_loss = 0
        steps = 0
        
        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tax_vector = batch['tax_vector'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                logits = model(input_ids, attention_mask, tax_vector)
                loss = criterion(logits, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            steps += 1
            pbar.set_postfix({'loss': total_loss/steps})
            
        # Step scheduler after EACH EPOCH
        scheduler.step()
            
        train_loss = total_loss / steps
        print(f"Epoch {epoch} Train Loss: {train_loss:.4f}, LR: {current_lr:.2e}")
        
        # Validation Loss Check
        val_loss = validate_loss(model, val_loader, criterion, device)
        print(f"Epoch {epoch} Val Loss: {val_loss:.4f}")
        
        epoch_time = time.time() - epoch_start_time
        
        # Log to MLflow
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
            "epoch_time": epoch_time
        }, step=epoch)
        
        if val_loss < best_val_loss:
            print(f"New Best Val Loss: {val_loss:.4f} (was {best_val_loss:.4f})")
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, {'val_loss': val_loss}, best_model_path)
            mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)
        
        # Custom Evaluation Schedule: 3, 10, 15, 20
        if epoch in [3, 10, 15, 20]:
            print(f"Epoch {epoch}: Running CAFA Evaluation on Best Model (Loss: {best_val_loss:.4f})...")
            
            # Save current state to restore after eval
            current_state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict() 
            }
            
            # Load best model
            if os.path.exists(best_model_path):
                checkpoint = torch.load(best_model_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded best model from epoch {checkpoint['epoch']} for evaluation.")
            else:
                print("Warning: Best model not found, evaluating current model.")

            # Run Evaluation (GPU Optimized)
            df_res = None
            # Run Evaluation (GPU Optimized)
            df_res = None
            pred_file = output_dir / f"gpu_preds_epoch_{epoch}.tsv"
            metrics_file = output_dir / f"evaluation_metrics_epoch_{epoch}.tsv"
            
            wf_max, best_thresh, _ = evaluate_gpu(
                model, val_loader, ic_weights, device,
                pred_output_path=pred_file,
                metrics_output_path=metrics_file
            )
            print(f"GPU Evaluated Weighted F-max: {wf_max:.4f} (at threshold {best_thresh:.2f})")
            
            # Save legacy prediction file if needed? Maybe skip for performance.
            # Only creating minimal artifacts to save IO.
            
            if df_res is not None and not isinstance(df_res, dict):
                 # We don't have df_res anymore from evaluate_gpu
                 pass
                 
            # Log Evaluation Metrics if available
            if wf_max > 0:
                mlflow.log_metric("weighted_fmax", wf_max, step=epoch)
                print(f"Logged Weighted F-max: {wf_max:.4f}")
                
                # Update best F-max
                if wf_max > best_wf_max:
                    best_wf_max = wf_max
                    print(f"New Best Weighted F-max: {best_wf_max:.4f}")
                    # Save best F-max model
                    save_checkpoint(model, optimizer, epoch, {'val_loss': best_val_loss, 'wf_max': best_wf_max}, output_dir / "best_model_fmax.pth")
                    with open(output_dir / "evaluation_best_weighted_fmax.tsv", "w") as f:
                        f.write(f"epoch\t{epoch}\tfmax\t{best_wf_max}\tthreshold\t{best_thresh}\n")
            
            # Restore current state to continue training
            model.load_state_dict(current_state['model'])
            # Optimizer state doesn't need restore if we didn't step, but good practice
            optimizer.load_state_dict(current_state['optimizer'])
            print("Restored training state.")

        save_checkpoint(model, optimizer, epoch, {'val_loss': val_loss}, output_dir / "latest_model.pth")
    
    mlflow.end_run()

if __name__ == "__main__":
    main()
