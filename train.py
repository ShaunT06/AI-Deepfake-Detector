import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time

# 1. Hardware Acceleration (Optimized for your i7/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training Engine started on: {device}")

# 2. Data Augmentation (This makes the model much more "Impressive")
# We flip and rotate images so the model learns "Features", not just static pixels.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Load Dataset
train_dataset = datasets.ImageFolder('Dataset/Train', transform=transform)
val_dataset = datasets.ImageFolder('Dataset/Validation', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 4. Model Surgery (Transfer Learning)
model = models.resnet18(weights='IMAGENET1K_V1')

# Freeze the "Backbone" to keep pretrained knowledge
for param in model.parameters():
    param.requires_grad = False

# Add a sophisticated "Classifier Head"
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.4), # Prevents cheating/overfitting
    nn.Linear(256, 2) # [Real, Fake]
)
model = model.to(device)

# 5. Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 6. The Training Loop (The Impressive Part)
epochs = 5
print(f"Starting Training for {epochs} epochs...")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_acc = 100. * correct / total
    print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.3f} | Acc: {epoch_acc:.2f}% | Time: {time.time()-start_time:.1f}s")

# 7. Final Save
torch.save(model.state_dict(), "deepfake_model.pth")
print("✅ MODEL TRAINED & SAVED. Ready for Grad-CAM integration.")