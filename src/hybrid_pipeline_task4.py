import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import label, regionprops_table

from unet_task3 import SmallUNet
from classical_task2 import query_text_llm

OUTPUT_DIR = os.path.abspath("outputs/task4_hybrid")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_trained_unet(model_path, device='cpu'):
    model = SmallUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def run_hybrid_pipeline():
    print("=== Running Task 4: Full Hybrid Pipeline on Unseen Test Images ===")
    dataset_dir = os.path.abspath('assignment3_dataset/nuclei_dataset')
    test_img_dir = os.path.join(dataset_dir, 'test', 'images')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model_path = os.path.abspath('outputs/task3_unet/unet_bce_dice.pth')
    
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint {model_path} not found. Ensure Task 3 completes first.")
        return
        
    unet_model = load_trained_unet(model_path, device=device)
    
    test_files = sorted([f for f in os.listdir(test_img_dir) if f.endswith('.png')])
    print(f"Processing {len(test_files)} unseen test images...")
    
    pipeline_records = []
    
    for fname in test_files:
        img_id = fname.replace('.png', '')
        img_path = os.path.join(test_img_dir, fname)
        
        # 1. Load and preprocess raw image
        img = Image.open(img_path).convert('L').resize((256, 256), Image.Resampling.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        # 2. U-Net Segmentation Mask prediction
        with torch.no_grad():
            logits = unet_model(img_tensor)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_mask = (prob > 0.5).astype(np.uint8)
            
        # 3. Connected components & Regionprops Extraction
        labeled_mask = label(pred_mask)
        props = regionprops_table(
            labeled_mask,
            intensity_image=img_np,
            properties=('label', 'area', 'eccentricity', 'solidity', 'mean_intensity', 'extent')
        )
        df_props = pd.DataFrame(props)
        n_objects = len(df_props)
        
        if n_objects > 0:
            mean_area = float(df_props['area'].mean())
            mean_ecc = float(df_props['eccentricity'].mean())
            mean_sol = float(df_props['solidity'].mean())
            mean_int = float(df_props['mean_intensity'].mean())
        else:
            mean_area, mean_ecc, mean_sol, mean_int = 0.0, 0.0, 0.0, 0.0
            
        # Classify density regime based on object count range
        if n_objects < 15:
            density_class = "sparse"
        elif n_objects <= 35:
            density_class = "normal"
        elif n_objects <= 60:
            density_class = "dense"
        else:
            density_class = "clustered"
            
        quality_flag = "good"
        
        # 4. Construct Source of Truth JSON record
        json_record = {
            "image_id": img_id,
            "n_objects": n_objects,
            "mean_area": round(mean_area, 2),
            "mean_eccentricity": round(mean_ecc, 3),
            "mean_solidity": round(mean_sol, 3),
            "mean_intensity": round(mean_int, 3),
            "density_class": density_class,
            "quality_flag": quality_flag
        }
        
        # 5. Generate Constrained One-Paragraph Narrative via Text LLM
        summary_text = (
            f"Image ID: {img_id}\n"
            f"- Number of Objects (Nuclei): {n_objects}\n"
            f"- Mean Object Area: {mean_area:.2f} pixels\n"
            f"- Mean Eccentricity: {mean_ecc:.3f}\n"
            f"- Mean Solidity: {mean_sol:.3f}\n"
            f"- Mean Intensity: {mean_int:.3f}\n"
            f"- Density Classification: {density_class}\n"
            f"- Quality Flag: {quality_flag}"
        )
        
        narrative_response = query_text_llm(summary_text)
        json_record["narrative"] = narrative_response
        
        pipeline_records.append(json_record)
        print(f"[{img_id}] N={n_objects}, Mean Area={mean_area:.1f}, Density={density_class}")
        
    # Aggregate into DataFrame & Save CSV
    df_results = pd.DataFrame(pipeline_records)
    csv_out = os.path.join(OUTPUT_DIR, "test_pipeline_results.csv")
    df_results.to_csv(csv_out, index=False)
    print(f"\nSaved aggregated test pipeline DataFrame to {csv_out}")
    
    json_out = os.path.join(OUTPUT_DIR, "test_pipeline_records.json")
    with open(json_out, "w") as f:
        json.dump(pipeline_records, f, indent=2)
    print(f"Saved test JSON records to {json_out}")

if __name__ == "__main__":
    run_hybrid_pipeline()
