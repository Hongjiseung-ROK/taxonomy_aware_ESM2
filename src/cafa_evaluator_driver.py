import argparse
import os
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Azure ML Driver for CAFA Evaluator")
    parser.add_argument("--data_dir", type=str, required=True, help="Mount point for the data asset")
    parser.add_argument("--prediction_path", type=str, required=True, help="Relative path to prediction files inside data asset")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory for results")
    args = parser.parse_args()

    # Define relative paths for assets within the mounted data directory
    # Updated based on user feedback:
    # ground truth: validation/validation_superset_term.tsv
    # obo: go_info/go-basic.obo
    # ia: IA.tsv
    
    gt_rel_path = "validation/validation_superset_term.tsv"
    obo_rel_path = "go_info/go-basic.obo"
    ia_rel_path = "IA.tsv"

    # Construct absolute paths
    data_dir = Path(args.data_dir)
    
    ground_truth_path = data_dir / gt_rel_path
    ontology_path = data_dir / obo_rel_path
    ia_path = data_dir / ia_rel_path
    prediction_dir = data_dir / args.prediction_path
    output_dir = Path(args.output_dir)

    # basic validation
    missing = []
    if not ground_truth_path.exists(): missing.append(f"Ground truth: {ground_truth_path}")
    if not ontology_path.exists(): missing.append(f"Ontology: {ontology_path}")
    if not ia_path.exists(): missing.append(f"IA file: {ia_path}")
    if not prediction_dir.exists(): missing.append(f"Predictions: {prediction_dir}")
    
    if missing:
        print("="*60)
        print("❌ Critical Input Files Missing:")
        for m in missing: print(f" - {m}")
        print("Listing data dir contents for debugging:")
        try:
             for root, dirs, files in os.walk(data_dir):
                level = root.replace(str(data_dir), '').count(os.sep)
                if level < 2:
                    indent = ' ' * 4 * (level)
                    print(f'{indent}{os.path.basename(root)}/')
                    for f in files: print(f'{indent}    {f}')
        except: pass
        print("="*60)
        sys.exit(1)

    # Ensure output directory exists (though CAFA evaluator might create it, good practice)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve script path
    # We assume this script is running from the root of the source directory where CAFA-evaluator-PK is located.
    script_path = Path("CAFA-evaluator-PK/src/cafaeval/__main__.py")
    if not script_path.exists():
         if Path("src/CAFA-evaluator-PK/src/cafaeval/__main__.py").exists():
             script_path = Path("src/CAFA-evaluator-PK/src/cafaeval/__main__.py")
         else:
             print(f"❌ Error: CAFA evaluator script not found at {script_path}")
             sys.exit(1)
    
    # Set PYTHONPATH to include the CAFA-evaluator source
    # The module is in CAFA-evaluator-PK/src
    cafa_src_path = script_path.parent.parent.resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cafa_src_path) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        str(script_path),
        str(ontology_path),
        str(prediction_dir),
        str(ground_truth_path),
        "-ia", str(ia_path),
        "-out_dir", str(output_dir),
        "-no_orphans", # often good practice for stable evaluation, can remove if not desired
        "-th_step", "0.01"
    ]

    print("="*60)
    print("🚀 Starting CAFA Evaluator Wrapper")
    print(f"📂 Predictions: {prediction_dir}")
    print(f"📂 Outputs: {output_dir}")
    print(f"🔧 PYTHONPATH: {env['PYTHONPATH']}")
    print(f"📜 Command: {' '.join(cmd)}")
    print("="*60)

    try:
        # Run subprocess and stream output
        # Using check=True to raise exception on non-zero exit code
        process = subprocess.run(
            cmd,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr, # Pipe stderr to sys.stderr so it shows up in AML logs
            text=True,
            env=env # Pass modified env
        )
        print("="*60)
        print("✅ CAFA Evaluator completed successfully.")
        print("="*60)

    except subprocess.CalledProcessError as e:
        print("="*60)
        print("❌ CAFA EVALUATOR ERROR")
        print(f"Command failed with exit code {e.returncode}")
        print("See stderr above for details.")
        print("="*60)
        sys.exit(e.returncode)
    except Exception as e:
        print("="*60)
        print(f"❌ UNEXPECTED ERROR: {e}")
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    main()
