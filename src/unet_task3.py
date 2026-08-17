import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

OUTPUT_DIR = os.path.abspath("outputs/task3_unet")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. Dataset Definition
# ---------------------------------------------------------
class NucleiDataset(Dataset):
    def __init__(self, data_dir, split='train', target_size=(256, 256)):
        self.split_dir = os.path.join(data_dir, split)
        self.img_dir = os.path.join(self.split_dir, 'images')
        self.mask_dir = os.path.join(self.split_dir, 'masks')
        self.target_size = target_size
        
        self.filenames = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.png')])
        
    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = os.path.join(self.img_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)
        
        img = Image.open(img_path).convert('L').resize(self.target_size, Image.Resampling.BILINEAR)
        mask = Image.open(mask_path).convert('L').resize(self.target_size, Image.Resampling.NEAREST)
        
        img_arr = np.array(img, dtype=np.float32) / 255.0
        mask_arr = np.array(mask, dtype=np.float32) / 255.0
        mask_arr = (mask_arr > 0.5).astype(np.float32)
        
        img_tensor = torch.tensor(img_arr, dtype=torch.float32).unsqueeze(0) # [1, H, W]
        mask_tensor = torch.tensor(mask_arr, dtype=torch.float32).unsqueeze(0) # [1, H, W]
        
        return img_tensor, mask_tensor, fname

# ---------------------------------------------------------
# 2. Small U-Net Architecture (PyTorch)
# ---------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class SmallUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[16, 32, 64, 128]):
        super(SmallUNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        curr_in = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_in, feature))
            curr_in = feature
            
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # Decoder
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))
            
        # Output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_x)
            
        return self.final_conv(x)

# ---------------------------------------------------------
# 3. Loss Functions & Evaluation Metrics
# ---------------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice

class CombinedBCEDiceLoss(nn.Module):
    def __init__(self):
        super(CombinedBCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        
    def forward(self, logits, targets):
        return self.bce(logits, targets) + self.dice(logits, targets)

def compute_metrics(preds, targets, threshold=0.5, smooth=1e-6):
    binary_preds = (preds > threshold).float()
    
    intersection = (binary_preds * targets).sum(dim=(1, 2, 3))
    total_pred = binary_preds.sum(dim=(1, 2, 3))
    total_target = targets.sum(dim=(1, 2, 3))
    
    dice = (2.0 * intersection + smooth) / (total_pred + total_target + smooth)
    union = total_pred + total_target - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return dice.mean().item(), iou.mean().item()

# ---------------------------------------------------------
# 4. Training Loop
# ---------------------------------------------------------
def train_unet(dataset_dir, loss_type='bce_dice', epochs=5, lr=3e-3, batch_size=16, device='cpu'):
    print(f"\n--- Training U-Net with Loss: {loss_type.upper()} on {device} ---", flush=True)
    
    train_dataset = NucleiDataset(dataset_dir, split='train')
    val_dataset = NucleiDataset(dataset_dir, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = SmallUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    if loss_type == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif loss_type == 'dice':
        criterion = DiceLoss()
    elif loss_type == 'bce_dice':
        criterion = CombinedBCEDiceLoss()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
        
    history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_iou': []}
    
    best_val_dice = 0.0
    best_model_weights = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            
        epoch_train_loss = running_loss / len(train_dataset)
        
        # Validation
        model.eval()
        running_val_loss = 0.0
        val_dices, val_ious = [], []
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss = criterion(logits, masks)
                running_val_loss += loss.item() * imgs.size(0)
                
                probs = torch.sigmoid(logits)
                d, i = compute_metrics(probs, masks)
                val_dices.append(d)
                val_ious.append(i)
                
        epoch_val_loss = running_val_loss / len(val_dataset)
        mean_val_dice = np.mean(val_dices)
        mean_val_iou = np.mean(val_ious)
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_dice'].append(mean_val_dice)
        history['val_iou'].append(mean_val_iou)
        
        if mean_val_dice > best_val_dice:
            best_val_dice = mean_val_dice
            best_model_weights = model.state_dict()
            
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Dice: {mean_val_dice:.4f} | Val IoU: {mean_val_iou:.4f}", flush=True)
            
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        
    return model, history

def run_task3_and_ablation():
    print("=== Running Task 3: U-Net Training & Loss Ablation ===", flush=True)
    dataset_dir = os.path.abspath('assignment3_dataset/nuclei_dataset')
    device = torch.device('cpu') # Fast CPU execution to avoid MPS memory lock with Ollama
    print(f"Using compute device: {device}", flush=True)
    
    loss_types = ['bce', 'dice', 'bce_dice']
    all_histories = {}
    all_models = {}
    
    metrics_summary = []
    
    for loss_name in loss_types:
        model, hist = train_unet(dataset_dir, loss_type=loss_name, epochs=5, lr=3e-3, device=device)
        all_histories[loss_name] = hist
        all_models[loss_name] = model
        
        best_dice = max(hist['val_dice'])
        best_iou = max(hist['val_iou'])
        
        metrics_summary.append({
            'model': f'U-Net ({loss_name.upper()})',
            'val_dice': round(best_dice, 4),
            'val_iou': round(best_iou, 4)
        })
        
        ckpt_path = os.path.join(OUTPUT_DIR, f"unet_{loss_name}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}", flush=True)
        
    # Save quantitative metrics summary table
    df_metrics = pd.DataFrame(metrics_summary)
    metrics_out = os.path.join(OUTPUT_DIR, "unet_loss_ablation_metrics.csv")
    df_metrics.to_csv(metrics_out, index=False)
    print(f"Saved metrics summary table to {metrics_out}\n{df_metrics}\n", flush=True)
    
    # 1. Plot Loss & Dice Curves Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for loss_name in loss_types:
        axes[0].plot(all_histories[loss_name]['val_loss'], label=f"{loss_name.upper()} (Val Loss)")
    axes[0].set_title("Validation Loss Across Epochs", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    for loss_name in loss_types:
        axes[1].plot(all_histories[loss_name]['val_dice'], label=f"{loss_name.upper()} (Val Dice)")
    axes[1].set_title("Validation Dice Score Across Epochs", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Score")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    curve_out = os.path.join(OUTPUT_DIR, "unet_training_curves.png")
    plt.savefig(curve_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved U-Net training curves to {curve_out}", flush=True)
    
    # 2. Qualitative Visual Comparison Panels (Input | Ground Truth Mask | Otsu Mask | Best U-Net)
    best_model = all_models['bce_dice']
    best_model.eval()
    
    val_dataset = NucleiDataset(dataset_dir, split='val')
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    plt.suptitle("Validation Segmentation Comparison (Input | Ground Truth | Otsu | U-Net BCE+Dice)", fontsize=16, fontweight='bold')
    
    sample_indices = [0, 5, 10, 15]
    
    for row_idx, val_idx in enumerate(sample_indices):
        img_t, mask_t, fname = val_dataset[val_idx]
        img_np = img_t.squeeze().numpy()
        gt_mask_np = mask_t.squeeze().numpy()
        
        t_otsu = threshold_otsu(img_np)
        otsu_mask = (img_np > t_otsu).astype(np.float32)
        
        with torch.no_grad():
            logits = best_model(img_t.unsqueeze(0).to(device))
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            unet_mask = (prob > 0.5).astype(np.float32)
            
        d_otsu, i_otsu = compute_metrics(torch.tensor(otsu_mask).unsqueeze(0).unsqueeze(0), mask_t.unsqueeze(0))
        d_unet, i_unet = compute_metrics(torch.tensor(unet_mask).unsqueeze(0).unsqueeze(0), mask_t.unsqueeze(0))
        
        axes[row_idx, 0].imshow(img_np, cmap='gray')
        axes[row_idx, 0].set_title(f"Input ({fname})\nRaw Image", fontsize=10)
        axes[row_idx, 0].axis('off')
        
        axes[row_idx, 1].imshow(gt_mask_np, cmap='gray')
        axes[row_idx, 1].set_title("Ground Truth\nMask", fontsize=10)
        axes[row_idx, 1].axis('off')
        
        axes[row_idx, 2].imshow(otsu_mask, cmap='gray')
        axes[row_idx, 2].set_title(f"Otsu Classical\nDice: {d_otsu:.3f} | IoU: {i_otsu:.3f}", fontsize=10)
        axes[row_idx, 2].axis('off')
        
        axes[row_idx, 3].imshow(unet_mask, cmap='gray')
        axes[row_idx, 3].set_title(f"U-Net (BCE+Dice)\nDice: {d_unet:.3f} | IoU: {i_unet:.3f}", fontsize=10)
        axes[row_idx, 3].axis('off')
        
    plt.tight_layout()
    panel_out = os.path.join(OUTPUT_DIR, "unet_validation_panels.png")
    plt.savefig(panel_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved validation comparison panels to {panel_out}", flush=True)

if __name__ == "__main__":
    run_task3_and_ablation()
