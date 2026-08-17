import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

DATASET_DIR = os.path.abspath('assignment3_dataset/nuclei_dataset')
OUTPUT_DIR = os.path.abspath('outputs/task1_eda')

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_preprocess_image(img_path, target_size=(256, 256)):
    """
    Loads an image, converts it to grayscale, resizes to target_size,
    and normalizes pixel values to [0, 1].
    """
    img = Image.open(img_path).convert('L') # Convert RGB to grayscale
    img_resized = img.resize(target_size, Image.Resampling.BILINEAR)
    img_arr = np.array(img_resized, dtype=np.float32) / 255.0
    return img_arr

def run_eda():
    print("=== Running Task 1: Exploratory Data Analysis (EDA) ===")
    
    metadata_path = os.path.join(DATASET_DIR, 'metadata.csv')
    df_meta = pd.read_csv(metadata_path)
    print(f"Loaded metadata with {len(df_meta)} records.")
    print("Regime distribution:\n", df_meta['density'].value_counts())
    
    # 1. Select representative samples across 4 density regimes (sparse, normal, dense, clustered)
    regimes = ['sparse', 'normal', 'dense', 'clustered']
    sample_images = []
    
    for r in regimes:
        sample_row = df_meta[df_meta['density'] == r].iloc[0]
        split = sample_row['split']
        img_id = sample_row['image_id']
        img_file = f"{img_id}.png"
        
        img_path = os.path.join(DATASET_DIR, split, 'images', img_file)
        mask_path = os.path.join(DATASET_DIR, split, 'masks', img_file)
        
        img_arr = load_and_preprocess_image(img_path)
        mask_img = Image.open(mask_path).convert('L').resize((256, 256), Image.Resampling.NEAREST)
        mask_arr = np.array(mask_img, dtype=np.float32) / 255.0
        
        sample_images.append({
            'regime': r,
            'image_id': img_id,
            'n_objects': sample_row['n_objects'],
            'img_arr': img_arr,
            'mask_arr': mask_arr,
            'img_path': img_path
        })
        
    # Plot 1: Sample Grid across regimes
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    plt.suptitle("Fluorescence Microscopy Nuclei Dataset - Density Regimes", fontsize=16, fontweight='bold')
    
    for idx, sample in enumerate(sample_images):
        # Image
        ax_img = axes[0, idx]
        ax_img.imshow(sample['img_arr'], cmap='gray')
        ax_img.set_title(f"{sample['regime'].capitalize()} Regime\n({sample['image_id']}: N={sample['n_objects']})", fontsize=12)
        ax_img.axis('off')
        
        # Mask
        ax_mask = axes[1, idx]
        ax_mask.imshow(sample['mask_arr'], cmap='gray')
        ax_mask.set_title(f"Ground Truth Mask\n({sample['image_id']})", fontsize=12)
        ax_mask.axis('off')
        
    plt.tight_layout()
    grid_out = os.path.join(OUTPUT_DIR, 'eda_sample_grid.png')
    plt.savefig(grid_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved EDA sample grid to {grid_out}")
    
    # 2. Intensity Histogram Analysis
    all_intensities = []
    train_img_dir = os.path.join(DATASET_DIR, 'train', 'images')
    for img_fname in os.listdir(train_img_dir):
        if img_fname.endswith('.png'):
            path = os.path.join(train_img_dir, img_fname)
            arr = load_and_preprocess_image(path)
            all_intensities.append(arr.flatten())
            
    all_intensities = np.concatenate(all_intensities)
    
    plt.figure(figsize=(8, 5))
    plt.hist(all_intensities, bins=100, color='navy', alpha=0.8, edgecolor='black', linewidth=0.5)
    plt.title("Pixel Intensity Distribution (Grayscale 256x256 Train Set)", fontsize=14, fontweight='bold')
    plt.xlabel("Normalized Intensity [0.0 - 1.0]", fontsize=12)
    plt.ylabel("Pixel Count", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    hist_out = os.path.join(OUTPUT_DIR, 'eda_intensity_histogram.png')
    plt.savefig(hist_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved EDA intensity histogram to {hist_out}")
    
    return sample_images

if __name__ == '__main__':
    run_eda()
