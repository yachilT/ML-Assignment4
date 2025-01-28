import scipy
import os

import torch
from sklearn.model_selection import train_test_split
import torchvision
from torch import nn
from torchvision import transforms

from FlowerDataset import FlowerDataset
from training import train_last_layer


def load_dataset(images_path, labels_path):
    # Step 3: Load the labels from the .mat file
    labels_data = scipy.io.loadmat(labels_path)

    # The labels are in 'labels' key of the .mat file
    flower_labels = labels_data['labels'].flatten() - 1  # Flatten to get a 1D array of labels
    print(f"Max label: {max(flower_labels)}, Min label: {min(flower_labels)}")

    # Step 4: Load the images from the 'flowers/jpg/' directory
    flower_images = []


    # Check if the dataset directory exists
    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Dataset directory {images_path} not found!")

    # Ensure images and labels are properly loaded
    for i in range(1, len(flower_labels) + 1):
        image_path = os.path.join(images_path, f'image_{i:05}.jpg')
        if os.path.exists(image_path):
            flower_images.append(image_path)
        else:
            print(f"Image not found: {image_path}")

    # Check if images and labels were loaded correctly
    print(f"Number of images loaded: {len(flower_images)}")
    print(f"Number of labels loaded: {len(flower_labels)}")

    # Ensure that there are images and labels before proceeding
    if len(flower_images) == 0 or len(flower_labels) == 0:
        raise ValueError("No images or labels found. Please check the dataset.")

    # Step 5: Split the dataset into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(flower_images, flower_labels, test_size=0.25, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_dataloader(X, y, transform, batch_size=32):
    dataset = FlowerDataset(X, y, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def load_vgg19(num_classes=102):
    vgg_model = torchvision.models.vgg19(pretrained=True)
    for param in vgg_model.parameters():
        param.requires_grad = False  # Freezing the base model

    # Modify the final layer to match the number of classes (102 flowers)
    vgg_model.classifier[-1] = nn.Linear(vgg_model.classifier[-1].in_features, num_classes)

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg_model.to(device)

    # Step 4: Image Preprocessing for YOLOv5 segmentation and VGG19 classification
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize for VGG19 input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return vgg_model, transform



def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset('flowers/jpg/', 'flowers/imagelabels.mat')
    vgg_model, transform = load_vgg19()

    train_loader = get_dataloader(X_train, y_train, transform)
    val_loader = get_dataloader(X_val, y_val, transform)
    test_loader = get_dataloader(X_test, y_test, transform)

    train_losses, val_losses, test_losses, train_accuracies, val_accuracies, test_accuracies = train_last_layer(vgg_model, train_loader, val_loader, test_loader, num_epochs=5, learning_rate=0.001)








if __name__ == '__main__':
    main()