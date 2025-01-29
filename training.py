import torch
from torch import optim, nn
from tqdm import tqdm
import matplotlib.pyplot as plt



def train_last_layer(model, train_loader, val_loader, test_loader, num_epochs=10, learning_rate=0.001, last_train=True):
    # Automatically detect the device (GPU or CPU)

    if torch.cuda.is_available():
        print("using cuda")
        proccessor = "cuda"
    else:
        print("using cpu :(")
        proccessor = "cpu"
    device = torch.device(proccessor)

    # Move the model to the selected device (GPU or CPU)
    model.to(device)

    # Set model to train mode
    model.train()

    # Set the optimizer to update only the last layer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Lists to store the values of accuracy and loss
    train_losses, val_losses, test_losses = [], [], []
    train_accuracies, val_accuracies, test_accuracies = [], [], []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Train the model
        for inputs, labels in tqdm(train_loader):
            # Move the data to the same device as the model
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

        # Calculate train accuracy
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                # Move the data to the same device as the model
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = 100 * correct_val / total_val
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)

        # Test phase
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                # Move the data to the same device as the model
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_accuracy = 100 * correct_test / total_test
        test_losses.append(test_loss / len(test_loader))
        test_accuracies.append(test_accuracy)

        # Print stats for the current epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.2f}%, "
              f"Validation Loss: {val_losses[-1]:.4f}, Validation Accuracy: {val_accuracies[-1]:.2f}%, "
              f"Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.2f}%")
    return train_losses, val_losses, test_losses, train_accuracies, val_accuracies, test_accuracies

def plot_loss_accuracy(train_losses, val_losses, test_losses, train_accuracies, val_accuracies, test_accuracies, learning_rate, batch_size):

    # Plot the loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss vs. Epoch\nLearning rate: {learning_rate:.4f}, batch size: {batch_size}')
    plt.legend()

    plt.savefig("/home/yachil/repos/ML-Assignment4/Loss.png")
    plt.show()

    # Plot the accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs. Epoch\nLearning rate: {learning_rate:.4f}, batch size: {batch_size}')
    plt.legend()

    plt.savefig("/home/yachil/repos/ML-Assignment4/Acc.png")
    plt.show()