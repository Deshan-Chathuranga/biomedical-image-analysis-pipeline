import os
import json
import requests
import numpy as np
import pandas as pd
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening, binary_closing, disk, remove_small_objects
from skimage.measure import label, regionprops_table

OLLAMA_TEXT_URL = "http://127.0.0.1:11434/api/generate"
TEXT_MODEL = "llama3.2" # or qwen2.5:3b / phi3:mini

OUTPUT_DIR = os.path.abspath("outputs/task2_classical")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_classical_segmentation(img_path):
    """
    Applies Otsu thresholding, morphological cleanup, labels connected components,
    and extracts regionprops feature table.
    """
    img = Image.open(img_path).convert('L').resize((256, 256), Image.Resampling.BILINEAR)
    img_arr = np.array(img, dtype=np.float32) / 255.0
    
    # 1. Otsu thresholding
    t_otsu = threshold_otsu(img_arr)
    binary_mask = img_arr > t_otsu
    
    # 2. Morphological cleanup
    # Remove small noise objects < 10 pixels
    cleaned_mask = remove_small_objects(binary_mask, min_size=10)
    cleaned_mask = binary_opening(cleaned_mask, disk(2))
    cleaned_mask = binary_closing(cleaned_mask, disk(2))
    
    # 3. Label connected components
    labeled_mask = label(cleaned_mask)
    
    # 4. Compute per-object feature table with regionprops_table
    props = regionprops_table(
        labeled_mask,
        intensity_image=img_arr,
        properties=('label', 'area', 'eccentricity', 'solidity', 'mean_intensity', 'extent', 'perimeter')
    )
    df_props = pd.DataFrame(props)
    
    return img_arr, cleaned_mask, labeled_mask, df_props, t_otsu

def generate_numbers_summary(df_props, t_otsu):
    """
    Converts feature table into a concise natural-language quantitative summary (numbers-only).
    """
    n_objects = len(df_props)
    if n_objects == 0:
        return "No objects detected in the image."
    
    mean_area = df_props['area'].mean()
    std_area = df_props['area'].std() if n_objects > 1 else 0.0
    min_area = df_props['area'].min()
    max_area = df_props['area'].max()
    
    mean_ecc = df_props['eccentricity'].mean()
    mean_sol = df_props['solidity'].mean()
    mean_int = df_props['mean_intensity'].mean()
    
    summary = (
        f"Quantitative Image Analysis Summary:\n"
        f"- Total Connected Components (Objects): {n_objects}\n"
        f"- Otsu Intensity Threshold: {t_otsu:.4f}\n"
        f"- Object Area (pixels): mean = {mean_area:.2f} ± {std_area:.2f} (min = {min_area}, max = {max_area})\n"
        f"- Shape Eccentricity (0 = circle, 1 = line): mean = {mean_ecc:.3f}\n"
        f"- Shape Solidity (ratio of area to convex hull area): mean = {mean_sol:.3f}\n"
        f"- Object Mean Pixel Intensity: {mean_int:.3f}\n"
    )
    return summary

def query_text_llm(summary_text, model=TEXT_MODEL):
    prompt = f"""You are a quantitative biomedical data interpreter. You are provided strictly with numerical spatial and morphological measurements from an image segmentation pipeline. You have NOT seen the original image.

Numerical Summary:
{summary_text}

Based ONLY on the numerical metrics above, provide:
1. A concise one-paragraph description interpreting object count, density, morphology, and intensity distribution.
2. A structured JSON record strictly following this format:
{{
  "n_objects": int,
  "density_class": "sparse | normal | dense | clustered",
  "shape_regularity": "high | moderate | low",
  "quality_flag": "good | degraded | uncertain",
  "numbers_first_narrative": "string"
}}
"""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2
        }
    }
    try:
        res = requests.post(OLLAMA_TEXT_URL, json=payload, timeout=60)
        if res.status_code == 200:
            return res.json().get("response", "")
        else:
            print(f"Ollama text API error: {res.status_code}")
            return None
    except Exception as e:
        print(f"Failed to query LLM: {e}")
        return None

def run_task2():
    print("=== Running Task 2: Classical Features & LLM Interpretation ===")
    dataset_dir = os.path.abspath('assignment3_dataset/nuclei_dataset')
    meta_path = os.path.join(dataset_dir, 'metadata.csv')
    df_meta = pd.read_csv(meta_path)
    
    rep_row = df_meta[df_meta['density'] == 'normal'].iloc[0]
    img_id = rep_row['image_id']
    img_path = os.path.join(dataset_dir, rep_row['split'], 'images', f"{img_id}.png")
    
    print(f"Processing representative image: {img_id}")
    img_arr, cleaned_mask, labeled_mask, df_props, t_otsu = process_classical_segmentation(img_path)
    
    print(f"Otsu threshold: {t_otsu:.4f}, Detected {len(df_props)} objects.")
    df_props.to_csv(os.path.join(OUTPUT_DIR, f"{img_id}_regionprops.csv"), index=False)
    
    numbers_summary = generate_numbers_summary(df_props, t_otsu)
    print("\nGenerated Numbers Summary:\n", numbers_summary)
    
    with open(os.path.join(OUTPUT_DIR, f"{img_id}_numbers_summary.txt"), "w") as f:
        f.write(numbers_summary)
        
    print("\nQuerying Text LLM with Numbers Summary...")
    llm_response = query_text_llm(numbers_summary)
    print(f"LLM Response:\n{llm_response}")
    
    with open(os.path.join(OUTPUT_DIR, f"{img_id}_llm_numbers_first_response.txt"), "w") as f:
        f.write(llm_response)
        
    print(f"Saved Task 2 results to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_task2()
