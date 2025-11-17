import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io, os, numpy as np
import traceback

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
# Default model path - update if you saved it under a different name
MODEL_PATH = 'tumor_resnet50.pth'
IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Make sure this order matches the folder order printed by train_resnet.py
CLASSES = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary"
}

# ------------------------------------------------------------------
# Model Definition (ResNet-50 with a custom head)
# ------------------------------------------------------------------
class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=4, hidden_units=512, use_pretrained=True):
        super(ResNet50Classifier, self).__init__()

        # Load ResNet-50 with ImageNet weights where possible, fall back if torch/torchvision differ
        try:
            if use_pretrained:
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                model = models.resnet50(weights=None)
        except Exception:
            # older torchvision API
            model = models.resnet50(pretrained=use_pretrained)

        # Freeze earlier layers by default (you may have fine-tuned layer4 during training)
        for name, param in model.named_parameters():
            param.requires_grad = False
            if name.startswith("layer4") or name.startswith("fc"):
                # allow layer4 and fc to be trainable if training script unfreezes them
                param.requires_grad = True

        # Replace fc head to match training script
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_units, num_classes)
        )

        self.model = model

    def forward(self, x):
        return self.model(x)

# ------------------------------------------------------------------
# Transformations (same normalization used in training)
# ------------------------------------------------------------------
data_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------------------------------------------------------
# Flask App Setup
# ------------------------------------------------------------------
app = Flask(__name__)
CORS(app)
model = None

def try_load_state_dict(target_model, path, device):
    """
    Loads a state dict safely. Attempts several fallbacks:
    1) load exactly as-is
    2) if keys are prefixed (e.g., saved from a wrapper), strip common prefixes
    3) finally load with strict=False to accept missing/unexpected keys and warn
    """
    state = torch.load(path, map_location=device)
    # if a checkpoint dict contains other keys (like {'model_state_dict': ...}), try to find it
    if isinstance(state, dict):
        # common checkpoint patterns: 'state_dict', 'model_state_dict', 'model'
        possible_keys = ['state_dict', 'model_state_dict', 'model', 'net']
        found = False
        for k in possible_keys:
            if k in state and isinstance(state[k], dict):
                state = state[k]
                found = True
                break
        # if still a dict of tensors (proper state_dict), ok; else assume it's correct already.

    # Try to load directly first
    try:
        target_model.load_state_dict(state)
        print("Loaded state_dict into model with strict=True")
        return True
    except Exception as e:
        print("Direct load_state_dict failed:", e)
        # Attempt to detect and strip a common prefix like 'module.' or 'model.' from keys
        new_state = {}
        for k, v in state.items():
            new_key = k
            if k.startswith('module.'):
                new_key = k[len('module.'):]
            if k.startswith('model.'):
                new_key = k[len('model.'):]
            new_state[new_key] = v
        try:
            target_model.load_state_dict(new_state)
            print("Loaded state_dict after stripping common prefixes (module./model.)")
            return True
        except Exception as e2:
            print("Load after stripping prefixes failed:", e2)
            # Last resort: try load with strict=False
            try:
                target_model.load_state_dict(state, strict=False)
                print("Loaded state_dict with strict=False (some keys missing/unused).")
                return True
            except Exception as e3:
                print("Final attempt (strict=False) also failed:", e3)
                return False

@app.before_request
def load_model():
    """Load the model into memory once (safe and robust)."""
    global model
    if model is None:
        print(f"Attempting to load ResNet-50 model from: {MODEL_PATH} on device: {device}")
        try:
            model = ResNet50Classifier(num_classes=len(CLASSES), hidden_units=512, use_pretrained=True)
            ok = try_load_state_dict(model.model, MODEL_PATH, device)
            if not ok:
                raise RuntimeError("Failed to load state_dict into model. See logs above for details.")
            model.to(device)
            model.eval()
            print("Model loaded successfully and set to eval mode.")
        except Exception as e:
            print("FATAL ERROR loading model:")
            traceback.print_exc()
            # avoid partial server start; exit so you fix the model file first
            os._exit(1)

@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "Model Ready", "device": str(device)})

# ------------------------------------------------------------------
# Prediction Endpoint
# ------------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    try:
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read()))

        # --- quick validation ---
        if img.mode not in ['L', 'RGB']:
            return jsonify({"error": "Please upload a valid brain MRI image."}), 400
        if img.width < 100 or img.height < 100:
            return jsonify({"error": "Image too small (min 100×100)."}), 400
        img_arr = np.array(img.convert('L'))
        mean_intensity = img_arr.mean()
        if mean_intensity < 5 or mean_intensity > 250:
            return jsonify({"error": "Image brightness abnormal; not an MRI?"}), 400

        # --- preprocess ---
        img = img.convert('RGB')
        img_tensor = data_transforms(img).unsqueeze(0).to(device)

        # --- predict ---
        with torch.no_grad():
            logits = model(img_tensor)            # [1, num_classes]
            probs = torch.softmax(logits, dim=1)[0]
            conf, pred_idx = torch.max(probs, dim=0)

        result_text = CLASSES.get(int(pred_idx.item()), "Unknown")

        return jsonify({
            "prediction": result_text,
            "prediction_index": int(pred_idx.item()),
            "confidence": float(conf.item() * 100.0),
            "all_confidences": {
                CLASSES[i]: float(probs[i].item() * 100.0)
                for i in range(len(CLASSES))
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "prediction": "ERROR (Check Server Logs)",
            "confidence": 0.0
        })

# ------------------------------------------------------------------
if __name__ == '__main__':
    # attempt to load immediately (useful when run directly)
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
