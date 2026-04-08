import torch
from transformers import ViTModel, ViTImageProcessor
import cv2
import numpy as np
import os
from tqdm import tqdm

# ===== Configuration =====
# Auto-detect dataset path — works on any machine
POSSIBLE_PATHS = [
    "/content/LLVIP",                           # Google Colab
    "./LLVIP",                                  # Same folder
    "../LLVIP",                                 # One level up
]

DATASET_PATH = None
for path in POSSIBLE_PATHS:
    if os.path.exists(path):
        DATASET_PATH = path
        break

if DATASET_PATH is None:
    print("⚠️  Dataset not found in default locations.")
    print("Please enter the full path to the LLVIP folder:")
    DATASET_PATH = input("DATASET_PATH: ").strip()

# Output folder = same folder as this script
OUTPUT_PATH = os.path.dirname(os.path.abspath(__file__))
NUM_IMAGES  = 100

print(f"✅ Dataset path : {DATASET_PATH}")
print(f"✅ Output path  : {OUTPUT_PATH}")

# ===== Load ViT Model =====
print("\nLoading ViT model...")
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model     = ViTModel.from_pretrained("google/vit-base-patch16-224")
model.eval()
print("Model loaded successfully ✓")

# ===== Feature Extraction Function =====
def extract_features(image_path):
    """
    Extracts a 768-dimensional feature vector from an image using ViT.
    The CLS token output represents the global image representation.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    features = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return features

# ===== Process Images =====
infrared_dir = os.path.join(DATASET_PATH, "infrared", "train")
visible_dir  = os.path.join(DATASET_PATH, "visible",  "train")

if not os.path.exists(infrared_dir) or not os.path.exists(visible_dir):
    raise FileNotFoundError(
        f"Could not find infrared/visible folders inside: {DATASET_PATH}\n"
        f"Make sure the dataset structure is: LLVIP/infrared/train and LLVIP/visible/train"
    )

image_files = sorted(os.listdir(infrared_dir))[:NUM_IMAGES]

infrared_features = []
visible_features  = []

print(f"\nProcessing {NUM_IMAGES} image pairs...")

for img_name in tqdm(image_files, desc="Extracting features"):
    # Extract infrared features
    ir_path  = os.path.join(infrared_dir, img_name)
    ir_feat  = extract_features(ir_path)
    infrared_features.append(ir_feat)

    # Extract visible features
    vis_path = os.path.join(visible_dir, img_name)
    vis_feat = extract_features(vis_path)
    visible_features.append(vis_feat)

# ===== Fuse & Save Feature Vectors =====
infrared_features = np.array(infrared_features)
visible_features  = np.array(visible_features)

# Feature-level fusion: concatenate IR and visible vectors
fused_features = np.concatenate([infrared_features, visible_features], axis=1)

np.save(os.path.join(OUTPUT_PATH, "infrared_features.npy"), infrared_features)
np.save(os.path.join(OUTPUT_PATH, "visible_features.npy"),  visible_features)
np.save(os.path.join(OUTPUT_PATH, "fused_features.npy"),    fused_features)

# ===== Results Summary =====
print("\n===== Feature Extraction Results =====")
print(f"Infrared features shape : {infrared_features.shape}  (100 images x 768 features)")
print(f"Visible features shape  : {visible_features.shape}  (100 images x 768 features)")
print(f"Fused features shape    : {fused_features.shape} (100 images x 1536 features)")
print(f"\nOutput saved to: {OUTPUT_PATH}")
print("Assignment 2 complete ✓")