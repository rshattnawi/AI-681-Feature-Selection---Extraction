import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# ===== Load saved feature vectors =====
OUTPUT_PATH = r"C:\Users\rshat\OneDrive\Masaüstü\UNI\Second2025\AI 681\asg2\cv_assignment2"
DATASET_PATH = r"C:\Users\rshat\Downloads\LLVIP"

infrared_features = np.load(os.path.join(OUTPUT_PATH, "infrared_features.npy"))
visible_features  = np.load(os.path.join(OUTPUT_PATH, "visible_features.npy"))
fused_features    = np.load(os.path.join(OUTPUT_PATH, "fused_features.npy"))

# ===== Figure 1: Input Images (Visible vs Infrared) =====
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle("Input Images: Visible vs Infrared Pairs", fontsize=16, fontweight='bold')

image_files = sorted(os.listdir(os.path.join(DATASET_PATH, "infrared", "train")))[:4]

for i, img_name in enumerate(image_files):
    # Load visible
    vis = cv2.imread(os.path.join(DATASET_PATH, "visible", "train", img_name))
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    # Load infrared (single channel for colormap)
    ir = cv2.imread(os.path.join(DATASET_PATH, "infrared", "train", img_name))
    ir = cv2.cvtColor(ir, cv2.COLOR_BGR2RGB)
    ir_gray = cv2.cvtColor(ir, cv2.COLOR_RGB2GRAY)

    axes[0, i].imshow(vis)
    axes[0, i].set_title(f"Visible #{i+1}", fontsize=10)
    axes[0, i].axis('off')

    axes[1, i].imshow(ir_gray, cmap='inferno')
    axes[1, i].set_title(f"Infrared #{i+1}", fontsize=10)
    axes[1, i].axis('off')

    # Fused = blend both
    blended = cv2.addWeighted(vis, 0.5, ir, 0.5, 0)
    axes[2, i].imshow(blended)
    axes[2, i].set_title(f"Fused #{i+1}", fontsize=10)
    axes[2, i].axis('off')

axes[0, 0].set_ylabel("Visible", fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel("Infrared", fontsize=12, fontweight='bold')
axes[2, 0].set_ylabel("Fused", fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "input_images.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Figure 1 saved ✓")

# ===== Figure 2: Feature Vectors Heatmap =====
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Output: Feature Vectors Visualization (First 10 Images)", fontsize=14, fontweight='bold')

im1 = axes[0].imshow(infrared_features[:10], aspect='auto', cmap='viridis')
axes[0].set_title("Infrared Features\n(10 x 768)", fontweight='bold')
axes[0].set_xlabel("Feature Dimensions (768)")
axes[0].set_ylabel("Images")
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(visible_features[:10], aspect='auto', cmap='viridis')
axes[1].set_title("Visible Features\n(10 x 768)", fontweight='bold')
axes[1].set_xlabel("Feature Dimensions (768)")
plt.colorbar(im2, ax=axes[1])

im3 = axes[2].imshow(fused_features[:10], aspect='auto', cmap='viridis')
axes[2].set_title("Fused Features\n(10 x 1536)", fontweight='bold')
axes[2].set_xlabel("Feature Dimensions (1536)")
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "feature_vectors.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Figure 2 saved ✓")

# ===== Figure 3: Feature Statistics =====
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Feature Statistics", fontsize=14, fontweight='bold')

axes[0].hist(infrared_features.flatten(), bins=50, color='orange', alpha=0.7, edgecolor='black')
axes[0].set_title("Infrared Feature Distribution")
axes[0].set_xlabel("Feature Value")
axes[0].set_ylabel("Frequency")

axes[1].hist(visible_features.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
axes[1].set_title("Visible Feature Distribution")
axes[1].set_xlabel("Feature Value")

axes[2].hist(fused_features.flatten(), bins=50, color='green', alpha=0.7, edgecolor='black')
axes[2].set_title("Fused Feature Distribution")
axes[2].set_xlabel("Feature Value")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "feature_statistics.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Figure 3 saved ✓")

print("\n===== All visualizations saved! =====")
print(f"Check your folder: {OUTPUT_PATH}")