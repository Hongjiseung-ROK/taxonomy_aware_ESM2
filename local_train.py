import os
import sys
import subprocess
import argparse

def main():
    # Define paths
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_DIR = os.path.join(ROOT_DIR, "src")
    DATASET_DIR = os.path.join(ROOT_DIR, "dataset")

    # Verify input directories
    if not os.path.exists(SRC_DIR):
        print(f"Error: Source directory not found at {SRC_DIR}")
        return
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory not found at {DATASET_DIR}")
        return

    print(f"Root Dir: {ROOT_DIR}")
    print(f"Src Dir:  {SRC_DIR}")
    print(f"Data Dir: {DATASET_DIR}")

    # Construct the command
    # We run from SRC_DIR to match Azure ML behavior and allow relative imports
    cmd = [
        sys.executable, "train.py",
        "--data_path", DATASET_DIR,
        "--epochs", "5",
        "--batch_size", "64",
        "--lr", "1e-4",
        "--min_lr", "5e-6",
        "--num_workers", "8",  # 0 for local windows debugging usually safer
        "--esm_model_name", "facebook/esm2_t6_8M_UR50D",
        "--use_lora", "True",
        "--lora_rank", "8",
        # Asymmetric Loss defaults
        "--gamma_neg", "4",
        "--gamma_pos", "0",
        "--clip", "0.05",
        
        # Absolute locations for output
        "--output_dir", os.path.join(ROOT_DIR, "outputs"),
        "--mlflow_dir", os.path.join(ROOT_DIR, "mlruns")
    ]

    print(f"Running command: {' '.join(cmd)}")
    print("-" * 50)

    # Run the training script
    try:
        # cwd=SRC_DIR is crucial for relative imports
        subprocess.run(cmd, cwd=SRC_DIR, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
