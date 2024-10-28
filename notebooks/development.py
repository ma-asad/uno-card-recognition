import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# CNN Model Definition
class UNOCardClassifier(nn.Module):
    def __init__(self, num_classes=24):
        super(UNOCardClassifier, self).__init__()
        # Use ResNet18 as base model with pretrained weights
        self.model = models.resnet18(pretrained=True)
        # Replace the final fully connected layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

# Custom Dataset Class
class UNODataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

# Data preprocessing and loading
def load_data(data_dir):
    image_paths = []
    labels = []
    label_map = {}
    
    # Get all folders (each representing a card class)
    classes = sorted(os.listdir(data_dir))
    for idx, class_name in enumerate(classes):
        label_map[class_name] = idx
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for image_name in os.listdir(class_dir):
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(class_dir, image_name))
                    labels.append(idx)
    
    return image_paths, labels, label_map

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    best_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {epoch_loss:.4f}')
        print(f'Validation Accuracy: {accuracy:.2f}%')
        
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), 'best_model.pth')
    
    return train_losses, val_accuracies

# Prediction function for single image
def predict_image(model, image_path, transform, device='cuda', label_map=None):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
        
    predicted_class = predicted.item()
    if label_map:
        # Reverse the label map to get class name
        label_map_rev = {v: k for k, v in label_map.items()}
        return label_map_rev[predicted_class]
    return predicted_class

# Camera prediction function
def predict_from_camera(model, transform, device='cuda', label_map=None):
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB and then to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Transform and predict
        image_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output.data, 1)
        
        predicted_class = predicted.item()
        if label_map:
            label_map_rev = {v: k for k, v in label_map.items()}
            predicted_class = label_map_rev[predicted_class]
            
        # Display prediction on frame
        cv2.putText(frame, f"Predicted: {predicted_class}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('UNO Card Classifier', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Main execution
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load and split data
    data_dir = "path_to_your_data_directory"  # Update this
    image_paths, labels, label_map = load_data(data_dir)
    
    # Split into train and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = UNODataset(train_paths, train_labels, transform)
    val_dataset = UNODataset(val_paths, val_labels, transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = UNOCardClassifier(num_classes=24).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_losses, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device
    )
    
    # Plot training results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.show()
    
    # Example usage for single image prediction
    test_image_path = "path_to_test_image.jpg"  # Update this
    prediction = predict_image(model, test_image_path, transform, device, label_map)
    print(f"Predicted card: {prediction}")
    
    # Example usage for camera prediction
    predict_from_camera(model, transform, device, label_map)

if __name__ == "__main__":
    main()