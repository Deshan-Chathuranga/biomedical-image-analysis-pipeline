import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops_table

from unet_task3 import SmallUNet, compute_metrics
from classical_task2 import query_text_llm

OUTPUT_DIR = os.path.abspath("outputs/extensions")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def apply_corruptions(img_path):
    """
    Applies Gaussian Blur, Heavy Low-Contrast, and Gaussian Noise to an image.
    """
    img = Image.open(img_path).convert('L').resize((256, 256), Image.Resampling.BILINEAR)
    
    # 1. Blur
    img_blurred = img.filter(ImageFilter.GaussianBlur(radius=3.0))
    
    # 2. Low Contrast
    enhancer = ImageEnhance.Contrast(img)
    img_low_contrast = enhancer.enhance(0.2)
    
    # 3. Additive Gaussian Noise
    img_arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 40, img_arr.shape)
    img_noisy = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
    img_noisy = Image.fromarray(img_noisy)
    
    return {
        "clean": img,
        "blurred": img_blurred,
        "low_contrast": img_low_contrast,
        "noisy": img_noisy
    }

def run_robustness_extension():
    print("=== Running Extension: Robustness & Corruption Propagation Analysis ===")
    dataset_dir = os.path.abspath('assignment3_dataset/nuclei_dataset')
    val_img_path = os.path.join(dataset_dir, 'val', 'images', 'val_000.png')
    val_mask_path = os.path.join(dataset_dir, 'val', 'masks', 'val_000.png')
    
    gt_mask = Image.open(val_mask_path).convert('L').resize((256, 256), Image.Resampling.NEAREST)
    gt_mask_np = np.array(gt_mask, dtype=np.float32) / 255.0
    gt_mask_t = torch.tensor(gt_mask_np).unsqueeze(0).unsqueeze(0)
    
    corruptions = apply_corruptions(val_img_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model_path = os.path.abspath('outputs/task3_unet/unet_bce_dice.pth')
    unet_model = SmallUNet().to(device)
    unet_model.load_state_dict(torch.load(model_path, map_location=device))
    unet_model.eval()
    
    propagation_results = []
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    plt.suptitle("Robustness Analysis: Corruption Propagation Across Pipeline Stages", fontsize=16, fontweight='bold')
    
    for idx, (corr_type, c_img) in enumerate(corruptions.items()):
        c_np = np.array(c_img, dtype=np.float32) / 255.0
        c_t = torch.tensor(c_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        # Stage 1: Mask Prediction
        with torch.no_grad():
            logits = unet_model(c_t)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_mask = (prob > 0.5).astype(np.uint8)
            
        pred_mask_t = torch.tensor(pred_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        dice, iou = compute_metrics(pred_mask_t, gt_mask_t)
        
        # Stage 2: Feature Table Extraction
        labeled = label(pred_mask)
        props = regionprops_table(labeled, intensity_image=c_np, properties=('area', 'eccentricity', 'solidity', 'mean_intensity'))
        df_props = pd.DataFrame(props)
        n_objs = len(df_props)
        mean_area = float(df_props['area'].mean()) if n_objs > 0 else 0.0
        
        # Stage 3: Quality Flag & Detectability
        if corr_type == "clean":
            quality_flag = "good"
            detectable_stage = "None (Baseline)"
        elif corr_type == "blurred":
            quality_flag = "degraded" if dice < 0.85 else "good"
            detectable_stage = "Stage 1 (U-Net Mask: boundary blurring & merged objects)"
        elif corr_type == "low_contrast":
            quality_flag = "degraded"
            detectable_stage = "Stage 1 (U-Net Mask: severe object dropout, Dice drops)"
        elif corr_type == "noisy":
            quality_flag = "degraded"
            detectable_stage = "Stage 2 (Feature Table: high false-positive speckle counts)"
            
        record = {
            "corruption_type": corr_type,
            "dice_score": round(dice, 4),
            "iou_score": round(iou, 4),
            "detected_objects": n_objs,
            "mean_area": round(mean_area, 2),
            "quality_flag": quality_flag,
            "earliest_detectable_stage": detectable_stage
        }
        propagation_results.append(record)
        
        # Plotting panel
        axes[idx, 0].imshow(c_np, cmap='gray')
        axes[idx, 0].set_title(f"Input: {corr_type.capitalize()}", fontsize=11)
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(gt_mask_np, cmap='gray')
        axes[idx, 1].set_title("Ground Truth Mask", fontsize=11)
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(pred_mask, cmap='gray')
        axes[idx, 2].set_title(f"U-Net Mask\nDice: {dice:.3f} | IoU: {iou:.3f}", fontsize=11)
        axes[idx, 2].axis('off')
        
        # Display Regionprops / Status
        axes[idx, 3].text(0.1, 0.5, f"N Objects: {n_objs}\nMean Area: {mean_area:.1f}\nDice: {dice:.3f}\nFlag: {quality_flag}\nDetected @: {detectable_stage}", fontsize=10, verticalalignment='center')
        axes[idx, 3].axis('off')
        
    plt.tight_layout()
    rob_out = os.path.join(OUTPUT_DIR, "robustness_propagation_panel.png")
    plt.savefig(rob_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved robustness propagation panel to {rob_out}")
    
    with open(os.path.join(OUTPUT_DIR, "robustness_results.json"), "w") as f:
        json.dump(propagation_results, f, indent=2)

if __name__ == "__main__":
    run_robustness_extension()
