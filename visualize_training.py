import numpy as np
import matplotlib.pyplot as plt
import os
from args import Config

def load_fold_data(fold_path):
    """Loads training, validation, and test metrics for a specific fold."""
    data = {}
    metrics = ['train_ACC', 'train_LOSS', 'val_ACC', 'val_LOSS', 'test_ACC', 'test_LOSS']
    for metric in metrics:
        file_path = os.path.join(fold_path, f'{metric}.npy')
        if os.path.exists(file_path):
            data[metric] = np.load(file_path)
        else:
            print(f'Warning: File not found - {file_path}')
            data[metric] = None # Handle missing files gracefully
    return data

def plot_metrics(fold_data, fold_index, save_dir='plots'):
    """Plots training and validation accuracy and loss for a given fold."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    epochs = range(1, len(fold_data.get('train_ACC', [])) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    if fold_data.get('train_ACC') is not None:
        plt.plot(epochs, fold_data['train_ACC'], 'b-', label='Training ACC')
    if fold_data.get('val_ACC') is not None:
        plt.plot(epochs, fold_data['val_ACC'], 'r-', label='Validation ACC')
    plt.title(f'Fold {fold_index} - Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    if fold_data.get('train_LOSS') is not None:
        plt.plot(epochs, fold_data['train_LOSS'], 'b-', label='Training LOSS')
    if fold_data.get('val_LOSS') is not None:
        plt.plot(epochs, fold_data['val_LOSS'], 'r-', label='Validation LOSS')
    plt.title(f'Fold {fold_index} - Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'fold_{fold_index}_metrics.png'))
    # plt.show() # Uncomment to display plots immediately
    plt.close() # Close the figure to free memory

def plot_average_metrics(all_folds_data, num_folds, num_epochs, save_dir='plots'):
    """Plots the average training and validation metrics across all folds."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    avg_train_acc = np.zeros(num_epochs)
    avg_val_acc = np.zeros(num_epochs)
    avg_train_loss = np.zeros(num_epochs)
    avg_val_loss = np.zeros(num_epochs)
    valid_folds_count = {'train_ACC': 0, 'val_ACC': 0, 'train_LOSS': 0, 'val_LOSS': 0}

    for fold_data in all_folds_data:
        if fold_data.get('train_ACC') is not None and len(fold_data['train_ACC']) == num_epochs:
            avg_train_acc += fold_data['train_ACC']
            valid_folds_count['train_ACC'] += 1
        if fold_data.get('val_ACC') is not None and len(fold_data['val_ACC']) == num_epochs:
            avg_val_acc += fold_data['val_ACC']
            valid_folds_count['val_ACC'] += 1
        if fold_data.get('train_LOSS') is not None and len(fold_data['train_LOSS']) == num_epochs:
            avg_train_loss += fold_data['train_LOSS']
            valid_folds_count['train_LOSS'] += 1
        if fold_data.get('val_LOSS') is not None and len(fold_data['val_LOSS']) == num_epochs:
            avg_val_loss += fold_data['val_LOSS']
            valid_folds_count['val_LOSS'] += 1

    # Avoid division by zero if no valid data for a metric
    if valid_folds_count['train_ACC'] > 0: avg_train_acc /= valid_folds_count['train_ACC']
    if valid_folds_count['val_ACC'] > 0: avg_val_acc /= valid_folds_count['val_ACC']
    if valid_folds_count['train_LOSS'] > 0: avg_train_loss /= valid_folds_count['train_LOSS']
    if valid_folds_count['val_LOSS'] > 0: avg_val_loss /= valid_folds_count['val_LOSS']

    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    # Plot Average Accuracy
    plt.subplot(1, 2, 1)
    if valid_folds_count['train_ACC'] > 0:
        plt.plot(epochs, avg_train_acc, 'b-', label=f'Avg Training ACC ({valid_folds_count["train_ACC"]}/{num_folds} folds)')
    if valid_folds_count['val_ACC'] > 0:
        plt.plot(epochs, avg_val_acc, 'r-', label=f'Avg Validation ACC ({valid_folds_count["val_ACC"]}/{num_folds} folds)')
    plt.title('Average Training and Validation Accuracy Across Folds')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot Average Loss
    plt.subplot(1, 2, 2)
    if valid_folds_count['train_LOSS'] > 0:
        plt.plot(epochs, avg_train_loss, 'b-', label=f'Avg Training LOSS ({valid_folds_count["train_LOSS"]}/{num_folds} folds)')
    if valid_folds_count['val_LOSS'] > 0:
        plt.plot(epochs, avg_val_loss, 'r-', label=f'Avg Validation LOSS ({valid_folds_count["val_LOSS"]}/{num_folds} folds)')
    plt.title('Average Training and Validation Loss Across Folds')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'average_metrics.png'))
    # plt.show() # Uncomment to display plots immediately
    plt.close()
    print(f"Average plots saved to '{os.path.join(save_dir, 'average_metrics.png')}'")

if __name__ == '__main__':
    config = Config()
    base_path = './Kfold_models'
    save_plot_dir = './training_plots'

    all_folds_data = []
    num_epochs = 0 # Determine from the first valid file

    for i in range(config.num_fold):
        fold_path = os.path.join(base_path, f'fold{i}')
        if os.path.isdir(fold_path):
            print(f'Processing {fold_path}...')
            fold_data = load_fold_data(fold_path)
            all_folds_data.append(fold_data)

            # Determine num_epochs from the first available data
            if num_epochs == 0:
                for key, data_array in fold_data.items():
                    if data_array is not None:
                        num_epochs = len(data_array)
                        print(f'Determined number of epochs: {num_epochs}')
                        break

            # Plot individual fold metrics if data exists
            if any(val is not None for val in fold_data.values()): # Check if any data was loaded
                 if len(fold_data.get('train_ACC', [])) > 0: # Check if data has content
                     plot_metrics(fold_data, i, save_dir=save_plot_dir)
                     print(f"Fold {i} plots saved to '{save_plot_dir}'")
                 else:
                     print(f'Skipping plot for Fold {i} due to empty data arrays.')
            else:
                print(f'Skipping plot for Fold {i} as no data files were found or loaded.')
        else:
            print(f'Directory not found: {fold_path}')

    # Plot average metrics if we have data and determined epochs
    if all_folds_data and num_epochs > 0:
        plot_average_metrics(all_folds_data, config.num_fold, num_epochs, save_dir=save_plot_dir)
    elif not all_folds_data:
        print("No fold data loaded. Cannot generate average plots.")
    else:
        print("Could not determine the number of epochs from data. Cannot generate average plots.")

    print("Visualization script finished.")