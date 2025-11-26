# ===================================================================
# train_resnet.py  -- switched to ResNet-50 for multiclass tumor detection
# ===================================================================
import os
import time
import copy
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ------------------ config ------------------
DATA_ROOT = 'C:/Users/Neeraj/Downloads/archive/'  # update as needed
BATCH_SIZE = 8         # ResNet-50 is larger. Adjust to your GPU memory.
NUM_EPOCHS = 5         # train longer when fine-tuning
IMG_SIZE = 224
HIDDEN_UNITS = 512     # suggestion: bigger head for ResNet-50
UNFREEZE_LAYER4 = True  # unfreeze layer4 for fine-tuning
USE_SAMPLER = False     # try True if classes are heavily imbalanced
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ----------------------------------------------

print("Using device:", DEVICE)

# ---------------- transforms ------------------
data_transforms = {
    'Training': transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05,0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'Testing': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# --------------- datasets/dataloaders --------------
image_datasets = {
    x: datasets.ImageFolder(os.path.join(DATA_ROOT, x), data_transforms[x])
    for x in ['Training', 'Testing']
}

class_names = image_datasets['Training'].classes
num_classes = len(class_names)
dataset_sizes = {x: len(image_datasets[x]) for x in ['Training', 'Testing']}

print("Classes (training folders order):", class_names)
print("Num classes:", num_classes)
print("Dataset sizes:", dataset_sizes)

# compute class counts & class weights
train_targets = image_datasets['Training'].targets  # list of labels
class_counts = Counter(train_targets)
counts_list = torch.tensor([class_counts.get(i, 0) for i in range(num_classes)], dtype=torch.float)
print("Class counts:", counts_list.tolist())

# avoid division by zero
class_weights = 1.0 / (counts_list + 1e-6)
# normalize
class_weights = class_weights / class_weights.sum() * num_classes
class_weights = class_weights.to(DEVICE)
print("Class weights (normalized):", class_weights.tolist())

# Weighted sampler (optional)
if USE_SAMPLER:
    # sample weight for each sample = weight of its class
    sample_weights = [class_weights[int(y)].item() for y in train_targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(image_datasets['Training'], batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
else:
    train_loader = DataLoader(image_datasets['Training'], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

dataloaders = {
    'Training': train_loader,
    'Testing': DataLoader(image_datasets['Testing'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
}

# ---------------- model --------------------------
def initialize_model(num_classes, hidden_units=HIDDEN_UNITS, unfreeze_layer4=UNFREEZE_LAYER4):
    """
    Initialize ResNet-50 with ImageNet weights. Replace final fc with custom head.
    Robust loading: tries the new torchvision Weights enum, falls back to pretrained=True.
    """
    # Try new torchvision enum-based weights first (torchvision >= 0.13+)
    try:
        model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    except Exception:
        # fallback for older torchvision versions
        model_ft = models.resnet50(pretrained=True)

    # Freeze all parameters first
    for name, param in model_ft.named_parameters():
        param.requires_grad = False

    # Unfreeze layer4 and fc if requested (fine-tune)
    if unfreeze_layer4:
        for name, param in model_ft.named_parameters():
            if name.startswith('layer4') or name.startswith('fc'):
                param.requires_grad = True

    # Replace classifier head
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, hidden_units),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(hidden_units, num_classes)
    )
    return model_ft.to(DEVICE)

model = initialize_model(num_classes=num_classes)

# prepare parameter groups: head params (higher lr) and layer4 (lower lr)
params_to_update = []
head_params = []
layer4_params = []

for name, param in model.named_parameters():
    if param.requires_grad:
        params_to_update.append(param)
        # note: fc parameters are under 'fc' (same as resnet18)
        if name.startswith('fc'):
            head_params.append(param)
        elif name.startswith('layer4'):
            layer4_params.append(param)

print("Trainable params count:", sum(p.numel() for p in params_to_update))
print("Head params count:", sum(p.numel() for p in head_params))
print("Layer4 params count:", sum(p.numel() for p in layer4_params))

# ---------------- loss / optimizer ----------------
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam([
    {'params': head_params, 'lr': 1e-3},
    {'params': layer4_params, 'lr': 1e-4}
], weight_decay=1e-4)

# Remove verbose argument (compatibility); keep ReduceLROnPlateau behavior
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# ---------------- training loop -------------------
def calc_confusion_matrix(preds, labels, n_classes):
    cm = torch.zeros((n_classes, n_classes), dtype=torch.int64)
    for p, t in zip(preds.view(-1), labels.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_va_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-'*20)
        epoch_cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

        for phase in ['Training', 'Testing']:
            if phase == 'Training':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            loader = dataloaders[phase]
            for inputs, labels in tqdm(loader, desc=f"{phase}"):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE, dtype=torch.long)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'Training'):
                    outputs = model(inputs)                     # logits [B, C]
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'Training':
                        loss.backward()
                        optimizer.step()

                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels).item()
                total_samples += batch_size

                # update confusion matrix
                epoch_cm += calc_confusion_matrix(preds.cpu(), labels.cpu(), num_classes)

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects / total_samples

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # summary per phase
            if phase == 'Testing':
                per_class_acc = []
                for i in range(num_classes):
                    true_i = epoch_cm[i, :].sum().item()
                    correct_i = epoch_cm[i, i].item()
                    acc_i = correct_i / true_i if true_i > 0 else 0.0
                    per_class_acc.append(acc_i)
                print("Confusion Matrix (rows=true, cols=pred):")
                print(epoch_cm.numpy())
                for i, acc_i in enumerate(per_class_acc):
                    print(f"  Class {i} ({class_names[i]}): acc = {acc_i:.4f}, support = {epoch_cm[i,:].sum().item()}")

                # scheduler step based on validation accuracy
                scheduler.step(epoch_acc)

                # save best
                if epoch_acc > best_va_acc:
                    best_va_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best val acc: {best_va_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model

# ---------------- run ----------------------------
if __name__ == "__main__":
    final_model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS)
    torch.save(final_model.state_dict(), 'tumor_resnet50.pth')
    print("Saved model to tumor_resnet50.pth")
