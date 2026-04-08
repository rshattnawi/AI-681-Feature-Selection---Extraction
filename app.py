from flask import Flask, request, jsonify, render_template
import torch
from transformers import ViTModel, ViTImageProcessor, ViTConfig
import cv2
import numpy as np
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# ===== Load ViT Model once at startup =====
print("Loading ViT model...")
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# Load with attentions enabled by default
config = ViTConfig.from_pretrained("google/vit-base-patch16-224")
config.output_attentions = True
model = ViTModel.from_pretrained("google/vit-base-patch16-224", config=config)
model.eval()
print("Model ready ✓")

# ===== Feature Extraction =====
def extract_features(img_array):
    """Extract 768-dim feature vector from image using ViT CLS token."""
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    inputs  = processor(images=img_rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    features = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return features

# ===== Attention Map =====
def generate_feature_map(img):
    """Generate attention heatmap overlay using ViT last-layer attention."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inputs  = processor(images=img_rgb, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Check attentions
    if outputs.attentions is None or len(outputs.attentions) == 0:
        return generate_simple_heatmap(img)

    try:
        # Last layer attention: (1, num_heads, seq_len, seq_len)
        # seq_len = 197 = 1 CLS + 196 patches
        last_attn = outputs.attentions[-1]  # (1, 12, 197, 197)

        # Average over heads → (197, 197)
        avg_attn = last_attn[0].mean(dim=0)  # (197, 197)

        # CLS token's attention to all patches (skip CLS itself)
        cls_attn = avg_attn[0, 1:]  # (196,)
        cls_attn = cls_attn.numpy()

        # Normalize
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)

        # Reshape to 14×14
        att_map = cls_attn.reshape(14, 14)

        # Resize to 224×224
        att_map_resized = cv2.resize(att_map, (224, 224))

        # Build colormap heatmap
        heatmap_colored = plt.cm.viridis(att_map_resized)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

        # Overlay on original
        img_small = cv2.resize(img_rgb, (224, 224))
        overlay   = cv2.addWeighted(img_small, 0.45, heatmap_colored, 0.55, 0)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
        fig.patch.set_facecolor('#0f0f1a')

        axes[0].imshow(att_map_resized, cmap='viridis')
        axes[0].set_title('Attention Heatmap', fontsize=9, color='white', pad=6)
        axes[0].axis('off')

        axes[1].imshow(overlay)
        axes[1].set_title('Overlay on Image', fontsize=9, color='white', pad=6)
        axes[1].axis('off')

        plt.tight_layout(pad=0.5)

        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg', bbox_inches='tight',
                    facecolor='#0f0f1a', dpi=120)
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    except Exception as e:
        print(f"Attention map error: {e}")
        return generate_simple_heatmap(img)

def generate_simple_heatmap(img):
    """Fallback: simple gradient-based visualization."""
    img_rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_small = cv2.resize(img_rgb, (224, 224))
    gray      = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)
    gray_norm = gray.astype(float) / 255.0

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    fig.patch.set_facecolor('#0f0f1a')
    axes[0].imshow(gray_norm, cmap='viridis')
    axes[0].set_title('Feature Map', fontsize=9, color='white')
    axes[0].axis('off')
    axes[1].imshow(img_small)
    axes[1].set_title('Input Image', fontsize=9, color='white')
    axes[1].axis('off')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', bbox_inches='tight',
                facecolor='#0f0f1a', dpi=120)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# ===== Routes =====
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/extract", methods=["POST"])
def extract():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file      = request.files["image"]
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img       = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # Extract feature vector
    features = extract_features(img)

    # Resize for preview
    h, w  = img.shape[:2]
    new_w = 300
    new_h = int(h * new_w / w)
    img_resized = cv2.resize(img, (new_w, new_h))

    _, buffer  = cv2.imencode(".jpg", img_resized)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    # Attention map
    feature_map_img = generate_feature_map(img)

    return jsonify({
        "features":      features.tolist(),
        "feature_count": len(features),
        "mean":          float(np.mean(features)),
        "std":           float(np.std(features)),
        "min":           float(np.min(features)),
        "max":           float(np.max(features)),
        "image_preview": img_base64,
        "feature_map":   feature_map_img,
        "input_shape":   f"{img.shape[1]} × {img.shape[0]} × {img.shape[2]}"
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)