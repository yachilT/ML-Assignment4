import pandas as pd
import matplotlib.pyplot as plt

def plot_results(csv_path):
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Check available columns
    print("Columns in CSV:", df.columns)
    
    # Create a figure with two subplots stacked vertically
    plt.figure(figsize=(8, 8))
    
    # Plot Loss
    plt.subplot(2, 1, 1)
    plt.plot(df['epoch'], df['train/loss'], label='Train Loss')
    plt.plot(df['epoch'], df['valid/loss'], label='Validation Loss')
    plt.plot(df['epoch'], df['test/loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Cross-Entropy-Loss')
    plt.grid()
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(2, 1, 2)
    plt.plot(df['epoch'], df['train/acc'], label='Train Accuracy')
    plt.plot(df['epoch'], df['valid/acc'], label='Validation Accuracy')
    plt.plot(df['epoch'], df['test'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.grid()
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("yolo.png")
    plt.show()

# Example usage
csv_path = 'yolov5/runs/train-cls/exp6/results.csv'
plot_results(csv_path)