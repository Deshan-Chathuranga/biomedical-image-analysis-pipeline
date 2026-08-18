import os
import json
import base64
import requests
import pandas as pd

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "llama3.2-vision"
ALT_MODEL_NAME = "llava:7b"

OUTPUT_DIR = os.path.abspath("outputs/vlm")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NAIVE_PROMPT = """Analyze this medical image in detail. Diagnose any abnormalities, pathology, or diseases present, and explain the diagnosis."""

STRUCTURED_PROMPT = """You are an objective biomedical image description system. Your sole task is to provide an objective visual description of the provided microscopy image. 
Do NOT provide any clinical diagnosis, disease classification, or treatment recommendation. If a property cannot be visually verified, explicitly write "uncertain".

Return your output strictly as a single raw valid JSON object without any Markdown formatting or extra text outside the JSON:
{
  "modality": "Fluorescence Microscopy",
  "tissue_type": "Stained Cell Nuclei",
  "notable_features": "describe object count range, spatial distribution, brightness, or clustering",
  "image_quality": "good",
  "descriptive_summary": "one-paragraph objective visual description"
}
"""

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def query_ollama_vlm(image_path, prompt, model=MODEL_NAME, temperature=0.7):
    b64_image = encode_image_to_base64(image_path)
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [b64_image],
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            print(f"Error from Ollama ({response.status_code}): {response.text}")
            return None
    except Exception as e:
        print(f"Request failed for model {model}: {e}")
        return None

def run_task1():
    print("=== Running Task 1: Multimodal LLM Description ===")
    dataset_dir = os.path.abspath('assignment3_dataset/nuclei_dataset')
    meta_path = os.path.join(dataset_dir, 'metadata.csv')
    df_meta = pd.read_csv(meta_path)
    
    rep_row = df_meta[df_meta['density'] == 'normal'].iloc[0]
    img_id = rep_row['image_id']
    img_path = os.path.join(dataset_dir, rep_row['split'], 'images', f"{img_id}.png")
    
    print(f"Selected representative image: {img_id} ({img_path})")
    
    # 1. Run Naive Prompt on llama3.2-vision
    print("\n--- 1. Testing Naive Prompt (llama3.2-vision) ---")
    naive_response = query_ollama_vlm(img_path, NAIVE_PROMPT, model=MODEL_NAME)
    if not naive_response:
        print(f"Fallback naive prompt to {ALT_MODEL_NAME}...")
        naive_response = query_ollama_vlm(img_path, NAIVE_PROMPT, model=ALT_MODEL_NAME)
        
    print(f"Naive Prompt Response:\n{naive_response}\n")
    with open(os.path.join(OUTPUT_DIR, "naive_prompt_response.txt"), "w") as f:
        f.write(f"Image ID: {img_id}\nPrompt: {NAIVE_PROMPT}\n\nResponse:\n{naive_response}")
        
    # 2. Run Structured Prompt on llama3.2-vision
    print("\n--- 2. Testing Structured Prompt (llama3.2-vision) ---")
    structured_response = query_ollama_vlm(img_path, STRUCTURED_PROMPT, model=MODEL_NAME)
    if not structured_response:
        print(f"Fallback structured prompt to {ALT_MODEL_NAME}...")
        structured_response = query_ollama_vlm(img_path, STRUCTURED_PROMPT, model=ALT_MODEL_NAME)
        
    print(f"Structured Prompt Response:\n{structured_response}\n")
    with open(os.path.join(OUTPUT_DIR, "structured_prompt_response.txt"), "w") as f:
        f.write(f"Image ID: {img_id}\nPrompt: {STRUCTURED_PROMPT}\n\nResponse:\n{structured_response or ''}")
        
    # 3. Non-determinism Evaluation (3 repeated runs)
    print("\n--- 3. Testing Non-Determinism Across 3 Repeated Runs ---")
    runs = []
    for i in range(1, 4):
        print(f"Run {i}...")
        resp = query_ollama_vlm(img_path, STRUCTURED_PROMPT, model=MODEL_NAME, temperature=0.7)
        if not resp:
            resp = query_ollama_vlm(img_path, STRUCTURED_PROMPT, model=ALT_MODEL_NAME, temperature=0.7)
        runs.append({"run": i, "response": resp})
        with open(os.path.join(OUTPUT_DIR, f"repeat_run_{i}.txt"), "w") as f:
            f.write(resp or "")
            
    summary_data = {
        "image_id": img_id,
        "naive_prompt": NAIVE_PROMPT,
        "naive_response": naive_response,
        "structured_prompt": STRUCTURED_PROMPT,
        "structured_response": structured_response,
        "repeat_runs": runs
    }
    
    with open(os.path.join(OUTPUT_DIR, "vlm_task1_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)
        
    print(f"Task 1 complete. Saved outputs to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_task1()
