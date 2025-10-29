# ==============================================================================
# 1. Setup and Imports
# ==============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import os
import time
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# 2. Data Loading and Preprocessing
# ==============================================================================

# IMPORTANT: UPDATE THIS PATH to where your 'Training' and 'Testing' folders are located.
# Example: If your data is in C:\Users\YourName\Desktop\archive, set DATA_ROOT to that path.
DATA_ROOT = 'C:/Users/Neeraj/Downloads/archive/' 

# Define data transformations. ResNet requires 224x224 input and ImageNet normalization.
data_transforms = {
    'Training': transforms.Compose([
        # Standard augmentation for training data
        transforms.RandomResizedCrop(224), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Standard ImageNet normalization for pre-trained models
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'Testing': transforms.Compose([
        # Consistent preprocessing for testing/validation
        transforms.Resize(256), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load datasets using ImageFolder
image_datasets = {
    x: datasets.ImageFolder(os.path.join(DATA_ROOT, x), data_transforms[x])
    for x in ['Training', 'Testing']
}

# Define dataloaders
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=0) 
    for x in ['Training', 'Testing']
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['Training', 'Testing']}
class_names = image_datasets['Training'].classes
num_classes = len(class_names) 

print(f"Classes: {class_names}")
print(f"Training samples: {dataset_sizes['Training']}")
print(f"Testing samples: {dataset_sizes['Testing']}")

# ==============================================================================
# 3. Model Definition (ResNet-18 Transfer Learning)
# ==============================================================================

def initialize_model():
    """Loads ResNet-18 and adapts the final layer for binary classification."""
    # Load the pre-trained ResNet-18 model
    model_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze all the parameters of the pre-trained model
    for param in model_ft.parameters():
        param.requires_grad = False

    # Get the number of input features for the final layer
    num_ftrs = model_ft.fc.in_features
    
    # Replace the final fully connected layer (fc)
    model_ft.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid() 
    )
    
    # Send the model to the target device
    model_ft = model_ft.to(device)
    
    return model_ft

# Initialize the model
model = initialize_model()

# Only the parameters of the new final layer are trainable
params_to_update = [param for param in model.parameters() if param.requires_grad]

print("Parameters being updated (only the final layer):", [name for name, param in model.named_parameters() if param.requires_grad])

# ==============================================================================
# 4. Loss Function and Optimizer
# ==============================================================================

# Use Binary Cross Entropy Loss (BCELoss) for single output + Sigmoid
criterion = nn.BCELoss()

# Optimizer only targets the parameters we want to update (the new fc layer)
optimizer = optim.Adam(params_to_update, lr=0.001)

# Learning Rate Scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# ==============================================================================
# 5. Training and Evaluation Function
# ==============================================================================

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=3):
    """Handles the training and validation loop."""
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['Training', 'Testing']:
            if phase == 'Training':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            # Use tqdm for a progress bar
            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase} Phase'):
                inputs = inputs.to(device)
                labels = labels.float().unsqueeze(1).to(device) 

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'Training'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    preds = (outputs > 0.5).long() 

                    # Backward pass and optimization only in training phase
                    if phase == 'Training':
                        loss.backward()
                        optimizer.step() # <<< optimizer.step() called FIRST
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data.long()).item()

            # --- Scheduler Step Correction ---
            # Call scheduler.step() AFTER the optimizer has performed its batch update.
            if phase == 'Training':
                scheduler.step() # <<< scheduler.step() called SECOND (after all batches)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save the best model based on validation accuracy
            if phase == 'Testing' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# ==============================================================================
# 6. Execute Training and 7. Save the Best Model
# ==============================================================================

if __name__ == '__main__':
    # Execute Training
    NUM_EPOCHS = 3 
    # Check if the data root exists before attempting to train
    if os.path.isdir(os.path.join(DATA_ROOT, 'Training')) and os.path.isdir(os.path.join(DATA_ROOT, 'Testing')):
        final_model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS)
        
        # Save the Best Model
        torch.save(final_model.state_dict(), 'tumor_resnet.pth')
        print("Model saved as tumor_resnet.pth")
    else:
        print("ERROR: Data directories not found. Please ensure DATA_ROOT is set correctly and contains 'Training' and 'Testing' folders.")
