import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset, random_split
import os
from PIL import Image
import matplotlib.pyplot as plt

# Define a custom dataset class to load and preprocess your data
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define a function to create the ResNet model
def create_resnet_model(num_classes):
    model = resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    epoch_losses = []  # To store the loss of each epoch
    
    print('Inside train model')
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_losses.append(epoch_loss)  # Append epoch loss to list
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # Validate the model
        val_loss, val_acc = validate_model(model, val_loader, criterion)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    print("Training complete.")
    return epoch_losses

def validate_model(model, val_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

    val_loss /= len(val_loader.dataset)
    val_acc = correct_preds / total_preds
    return val_loss, val_acc

def main():
    # Define the paths to your dataset
    fake_image_folder =  "/home/tr_swapna/Dhyanesh/Dataset/Train/Fake_12"
    real_image_folder = "/home/tr_swapna/Dhyanesh/Dataset/Train/Real_12"

    # Load your dataset and assign labels (0 for fake, 1 for real)
    fake_image_files = [os.path.join(fake_image_folder, f) for f in os.listdir(fake_image_folder)]
    real_image_files = [os.path.join(real_image_folder, f) for f in os.listdir(real_image_folder)]
    all_image_files = fake_image_files + real_image_files
    all_labels = [0] * len(fake_image_files) + [1] * len(real_image_files)

    # Define transformations for data augmentation and normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create custom dataset
    dataset = CustomDataset(all_image_files, all_labels, transform=transform)

    # Split dataset into train and validation sets (75:25)
    train_size = int(0.75 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create the ResNet model for binary classification (fake/real)
    num_classes = 2
    resnet_model = create_resnet_model(num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet_model.parameters(), lr=0.001)
    
    print("Executing the train_model")
    # Train the model
    epoch_losses = train_model(resnet_model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

    # Plot the training loss over epochs
    plt.plot(range(1, 11), epoch_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.show()

    # Save the trained model weights
    save_dir = "/home/tr_swapna/Dhyanesh/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "resnet_model.pth")
    torch.save(resnet_model.state_dict(), save_path)
    print(f"Model weights saved at: {save_path}")

if __name__ == "__main__":
    main()
