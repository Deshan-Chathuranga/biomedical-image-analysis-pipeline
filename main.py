import os
import sys
import time

print("=" * 70)
print("  COMPACT BIOMEDICAL IMAGE-ANALYSIS HYBRID AI PIPELINE")
print("=" * 70)

scripts = [
    ("Task 1: Exploratory Data Analysis & Preprocessing", "src/data_prep.py"),
    ("Task 1: Multimodal LLM Direct Description", "src/vlm_task1.py"),
    ("Task 2: Classical Features & LLM Interpretation", "src/classical_task2.py"),
    ("Task 3: PyTorch U-Net Segmentation & Loss Ablation", "src/unet_task3.py"),
    ("Task 4: Hybrid Auditable Pipeline on Test Images", "src/hybrid_pipeline_task4.py"),
    ("Extra Credit Extensions: Robustness Corruption Analysis", "src/extensions.py")
]

start_total = time.time()

for idx, (title, script_path) in enumerate(scripts, 1):
    print(f"\n[{idx}/{len(scripts)}] STEP: {title}")
    print("-" * 60)
    
    cmd = f"python3 -u {script_path}"
    ret = os.system(cmd)
    
    if ret != 0:
        print(f"\n❌ Error encountered during {title} (exit code {ret}). Stopping pipeline.")
        sys.exit(1)
        
    print(f"✅ Completed: {title}")

elapsed = time.time() - start_total
print("\n" + "=" * 70)
print(f"🎉 SUCCESS: Entire Biomedical AI Pipeline completed in {elapsed:.1f} seconds!")
print("Outputs saved to outputs/ directory.")
print("=" * 70)
