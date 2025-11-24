import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io, os, numpy as np
import traceback
import base64

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
MODEL_PATH = 'tumor_resnet50.pth'
IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        try:
            if use_pretrained:
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                model = models.resnet50(weights=None)
        except Exception:
            model = models.resnet50(pretrained=use_pretrained)

        for name, param in model.named_parameters():
            param.requires_grad = False
            if name.startswith("layer4") or name.startswith("fc"):
                param.requires_grad = True

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
# Grad-CAM helper (improved visual overlay)
# ------------------------------------------------------------------
def generate_gradcam_overlay(model_wrapper, input_tensor, target_index, orig_img):
    """
    Generate a Grad-CAM overlay image (PIL RGB) on top of the original MRI.
    - model_wrapper: ResNet50Classifier
    - input_tensor: [1,3,H,W] tensor (requires_grad=True) on device
    - target_index: int
    - orig_img: original PIL image (before transforms)
    """
    model_wrapper.eval()
    model = model_wrapper.model

    activations = []
    gradients = []

    target_layer = model.layer4

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # register hooks
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    # use full_backward_hook if available
    try:
        handle_bwd = target_layer.register_full_backward_hook(backward_hook)
    except AttributeError:
        handle_bwd = target_layer.register_backward_hook(backward_hook)

    # forward pass
    output = model_wrapper(input_tensor)
    idx = int(target_index)
    score = output[0, idx]

    # backward pass
    model_wrapper.zero_grad()
    model.zero_grad()
    score.backward(retain_graph=False)

    # remove hooks
    handle_fwd.remove()
    handle_bwd.remove()

    if not activations or not gradients:
        return None

    act = activations[0].detach()[0]   # [C,H,W]
    grad = gradients[0].detach()[0]    # [C,H,W]

    # global average pooling on gradients -> weights
    weights = grad.mean(dim=(1, 2))    # [C]

    cam = torch.zeros_like(act[0])
    for c, w in enumerate(weights):
        cam += w * act[c]

    cam = torch.relu(cam)
    cam_np = cam.cpu().numpy()

    if cam_np.max() <= 1e-6:
        return None

    # normalize to [0,1]
    cam_np = cam_np - cam_np.min()
    cam_np = cam_np / (cam_np.max() + 1e-6)

    # (optional) sharpen to emphasize high-activation regions
    cam_np = cam_np ** 1.5

    # keep only top 20% activations (everything else becomes near-zero)
    thr = np.percentile(cam_np, 80.0)
    cam_np = np.where(cam_np >= thr, cam_np, 0.0)

    # resize CAM to original image size
    H, W = orig_img.size[1], orig_img.size[0]   # PIL size is (W,H)
    cam_pil = Image.fromarray((cam_np * 255).astype(np.uint8), mode="L")
    cam_pil = cam_pil.resize((orig_img.size[0], orig_img.size[1]), resample=Image.BILINEAR)
    cam_np_resized = np.array(cam_pil, dtype=np.float32) / 255.0  # back to [0,1]

    # simple "jet-like" colormap for visualization
    v = cam_np_resized
    r = (255 * v).clip(0, 255)
    g = (255 * (1.0 - np.abs(v - 0.5) * 2.0)).clip(0, 255)  # high in mid-range
    b = (255 * (1.0 - v)).clip(0, 255)

    heatmap_rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)  # [H,W,3]

    # blend with original MRI
    orig_rgb = orig_img.convert("RGB").resize((orig_img.size[0], orig_img.size[1]))
    orig_arr = np.array(orig_rgb).astype(np.float32)

    alpha = 0.45  # overlay strength
    overlay = (alpha * heatmap_rgb + (1 - alpha) * orig_arr).clip(0, 255).astype(np.uint8)

    return Image.fromarray(overlay, mode="RGB")

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
    state = torch.load(path, map_location=device)
    if isinstance(state, dict):
        possible_keys = ['state_dict', 'model_state_dict', 'model', 'net']
        for k in possible_keys:
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break

    try:
        target_model.load_state_dict(state)
        print("Loaded state_dict into model with strict=True")
        return True
    except Exception as e:
        print("Direct load_state_dict failed:", e)
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
            try:
                target_model.load_state_dict(state, strict=False)
                print("Loaded state_dict with strict=False (some keys missing/unused).")
                return True
            except Exception as e3:
                print("Final attempt (strict=False) also failed:", e3)
                return False

@app.before_request
def load_model():
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
        except Exception:
            print("FATAL ERROR loading model:")
            traceback.print_exc()
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
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))

        # quick validation
        if img.mode not in ['L', 'RGB']:
            return jsonify({"error": "Please upload a valid brain MRI image."}), 400
        if img.width < 100 or img.height < 100:
            return jsonify({"error": "Image too small (min 100×100)."}), 400
        img_arr = np.array(img.convert('L'))
        mean_intensity = img_arr.mean()
        if mean_intensity < 5 or mean_intensity > 250:
            return jsonify({"error": "Image brightness abnormal; not an MRI?"}), 400

        # preprocess for model
        img_rgb = img.convert('RGB')
        img_tensor = data_transforms(img_rgb).unsqueeze(0).to(device)
        img_tensor.requires_grad_(True)

        # forward (no torch.no_grad, we need gradients)
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)

        result_text = CLASSES.get(int(pred_idx.item()), "Unknown")

        # Grad-CAM overlay
        heatmap_url = None
        try:
            overlay_pil = generate_gradcam_overlay(model, img_tensor, int(pred_idx.item()), img_rgb)
            if overlay_pil is not None:
                buf = io.BytesIO()
                overlay_pil.save(buf, format='PNG')
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode('utf-8')
                heatmap_url = f"data:image/png;base64,{b64}"
        except Exception:
            traceback.print_exc()
            heatmap_url = None

        response = {
            "prediction": result_text,
            "prediction_index": int(pred_idx.item()),
            "confidence": float(conf.item() * 100.0),
            "all_confidences": {
                CLASSES[i]: float(probs[i].item() * 100.0)
                for i in range(len(CLASSES))
            }
        }

        if heatmap_url is not None:
            response["heatmap"] = heatmap_url

        return jsonify(response)

    except Exception:
        traceback.print_exc()
        return jsonify({
            "prediction": "ERROR (Check Server Logs)",
            "confidence": 0.0
        })

# ------------------------------------------------------------------
if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
