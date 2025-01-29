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
    plt.plot(df['epoch'], df['test/loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(2, 1, 2)
    plt.plot(df['epoch'], df['metrics/accuracy_top5'], label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("yolo.png")
    plt.show()

# Example usage
csv_path = 'yolov5/runs/train-cls/exp3/results.csv'
plot_results(csv_path)