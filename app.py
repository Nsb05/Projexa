import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import os

# --- Configuration ---
# Use the working ResNet model path and file name
MODEL_PATH = 'tumor_resnet.pth' 
IMG_SIZE = 224
# Classes must match your training data (0=No Tumor, 1=Tumor)
CLASSES = {
    0: "No Tumor Detected",
    1: "Tumor Detected"
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Architecture (ResNet-18 for Transfer Learning) ---
class ResNetClassifier(nn.Module):
    """
    A transfer learning model using ResNet-18 backbone.
    This structure is designed to load weights saved from the training script
    where only the final FC layer was saved.
    """
    def __init__(self, num_classes=1):
        super(ResNetClassifier, self).__init__()
        # Load the pre-trained ResNet-18 model
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze all the parameters in the feature extraction layers
        for param in model.parameters():
            param.requires_grad = False
            
        # Replace the final fully-connected layer (fc) for binary classification
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
        )
        # Store the modified ResNet model as an attribute
        self.model = model 

    def forward(self, x):
        return self.model(x)

# --- Data Transformations (ImageNet Standard) ---
# MUST match the transformations used in your ResNet training script
data_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app) 
model = None

@app.before_request
def load_model():
    """Load the model before the first request is served."""
    global model
    if model is None:
        print(f"Attempting to load model from: {MODEL_PATH} on device: {device}")
        try:
            # Initialize model structure
            model = ResNetClassifier(num_classes=1)
            
            # Load the state dictionary. 
            state_dict = torch.load(MODEL_PATH, map_location=device)
            
            # Load the saved state dict into the model's internal ResNet structure.
            model.model.load_state_dict(state_dict) 
            
            model.to(device)
            model.eval() # Set model to evaluation mode
            print("Model loaded successfully.")
            
        except Exception as e:
            print(f"FATAL ERROR: An error occurred while loading the model structure or weights: {e}")
            os._exit(1)


@app.route('/status', methods=['GET'])
def status():
    """Simple health check endpoint for the frontend."""
    return jsonify({"status": "Model Ready", "device": str(device)})


@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    try:
        # 1. Image Processing
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img_tensor = data_transforms(img).unsqueeze(0).to(device)

        # 2. Prediction
        with torch.no_grad():
            output = model(img_tensor)
            
            # Apply Sigmoid activation to convert logit output to probability (0 to 1)
            probability = torch.sigmoid(output).item() 

        # 3. Interpretation and Confidence Calculation
        
        if probability >= 0.5:
            # Assuming Class 1 is Tumor (Positive Class)
            prediction = 1
            confidence = probability * 100
        else:
            # Assuming Class 0 is No Tumor (Negative Class)
            prediction = 0
            confidence = (1 - probability) * 100
        
        result_text = CLASSES.get(prediction, "Unknown")

        # 4. Response
        # Ensure confidence is a float and use standard keys
        return jsonify({
            "prediction": result_text,
            "confidence": float(confidence)
        })

    except Exception as e:
        # Log the full traceback error here for easier debugging
        import traceback
        print("-" * 50)
        print("PREDICTION RUNTIME ERROR:")
        traceback.print_exc()
        print("-" * 50)
        
        # --- CRITICAL FIX: Return 200 OK status to force JS into the success block ---
        # The frontend expects "prediction" (string) and "confidence" (number).
        # We send safe defaults to prevent the JavaScript error.
        return jsonify({
            "prediction": "ERROR (Check Server Logs)", 
            "confidence": 0.0 # Safe number type
        }) # Note: Removed 500 status code

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
