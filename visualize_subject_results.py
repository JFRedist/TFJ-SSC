import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from model import Transformer
from args import Config, Path
from data_loader import data_generator # Assuming data_generator loads all data


def get_subject_indices(path_labels, subject_id):
    """Find the start and end indices for a specific subject in the concatenated dataset."""
    label_files = sorted([f for f in os.listdir(path_labels) if f.endswith('_label.npy')])
    current_index = 0
    subject_indices = None
    subject_labels = None

    print(f"Searching for subject {subject_id} in label files...")
    for f in label_files:
        if subject_id in f:
            labels = np.load(os.path.join(path_labels, f))
            start_index = current_index
            end_index = current_index + len(labels)
            subject_indices = (start_index, end_index)
            subject_labels = labels
            print(f"Found {subject_id} in {f}. Index range: [{start_index}, {end_index}). Length: {len(labels)}")
            # Don't break here, continue counting indices for subsequent files
        
        # Update current_index even if it's not the target subject
        # This ensures correct indexing for subjects appearing later in the sorted list
        if subject_indices is None or f > f'{subject_id}_label.npy': # Optimization: only load length if needed
             labels = np.load(os.path.join(path_labels, f))
             current_index += len(labels)
        elif subject_indices is not None and f == f'{subject_id}_label.npy':
             current_index = subject_indices[1] # Already updated index when found

    if subject_indices:
        print(f"Total epochs counted up to the end of {subject_id}: {subject_indices[1]}")
    else:
        print(f"Subject {subject_id} not found in label files.")
        
    return subject_indices, subject_labels

def plot_hypnogram(y_true, y_pred, subject_id, fold_id, accuracy, f1, save_path):
    """Plot the ground truth and predicted hypnograms."""
    stages = ['W', 'N1', 'N2', 'N3/4', 'R']
    colors = ['lightgrey', 'lightblue', 'blue', 'darkblue', 'red']
    stage_map = {i: stages[i] for i in range(len(stages))}
    color_map = {i: colors[i] for i in range(len(stages))}

    fig, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
    epochs = np.arange(len(y_true))

    # Plot Ground Truth
    axes[0].plot(epochs, y_true, drawstyle='steps-post', color='black', linewidth=0.5)
    for i in range(len(stages)):
        axes[0].fill_between(epochs, y_true, where=(y_true == i), step='post', color=color_map[i], alpha=0.7)
    axes[0].set_yticks(np.arange(len(stages)))
    axes[0].set_yticklabels(stages)
    axes[0].set_ylim(-0.5, len(stages) - 0.5)
    axes[0].set_ylabel('Ground Truth')
    axes[0].set_title(f'Sleep Stages for Subject {subject_id} (Fold {fold_id}) - Ground Truth vs. Prediction\nAccuracy: {accuracy:.2%}, F1-Score: {f1:.4f}')
    axes[0].grid(True, axis='y', linestyle=':')

    # Plot Prediction
    axes[1].plot(epochs, y_pred, drawstyle='steps-post', color='black', linewidth=0.5)
    for i in range(len(stages)):
        axes[1].fill_between(epochs, y_pred, where=(y_pred == i), step='post', color=color_map[i], alpha=0.7)
    axes[1].set_yticks(np.arange(len(stages)))
    axes[1].set_yticklabels(stages)
    axes[1].set_ylim(-0.5, len(stages) - 0.5)
    axes[1].set_ylabel('Prediction')
    axes[1].set_xlabel('Epoch (30s)')
    axes[1].grid(True, axis='y', linestyle=':')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap
    
    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Hypnogram saved to: {save_path}")
    # plt.show()
    plt.close()

def main(config, path, subject_id, fold_id):
    print(f"Visualizing results for Subject: {subject_id}, Fold: {fold_id}")

    # 1. Load all data and labels (needed for KFold split)
    print("Loading dataset and labels...")
    # Assuming data_generator returns the full dataset and labels needed for split
    # We need the *order* preserved as per file concatenation
    dataset, labels, _ = data_generator(path_labels=path.path_labels, path_dataset=path.path_TF)
    print(f"Dataset shape: {dataset.shape}, Labels shape: {labels.shape}")

    # 2. Find indices for the target subject in the *original* concatenated list
    subject_global_indices, subject_true_labels = get_subject_indices(path.path_labels, subject_id)
    if subject_global_indices is None:
        return
    start_idx_global, end_idx_global = subject_global_indices

    # 3. Replicate K-Fold split to find test indices for the target fold
    kf = StratifiedKFold(n_splits=config.num_fold, shuffle=True, random_state=0)
    fold_count = 0
    test_indices_for_fold = None
    for _, test_idx in kf.split(dataset, labels):
        if fold_count == fold_id:
            test_indices_for_fold = test_idx
            print(f"Found test indices for Fold {fold_id}. Count: {len(test_indices_for_fold)}")
            break
        fold_count += 1

    if test_indices_for_fold is None:
        print(f"Error: Could not find test indices for Fold {fold_id}.")
        return

    # 4. Check if the subject is in the test set for this fold
    subject_indices_in_fold_test_set = [idx for idx in test_indices_for_fold if start_idx_global <= idx < end_idx_global]

    if not subject_indices_in_fold_test_set:
        print(f"Subject {subject_id} is NOT in the test set for Fold {fold_id}.")
        # Optional: List subjects that *are* in the test set
        # test_subjects = set()
        # label_files = sorted([f for f in os.listdir(path.path_labels) if f.endswith('_label.npy')])
        # current_idx = 0
        # for f in label_files:
        #     subj_id_from_file = f.split('_')[0]
        #     num_labels = len(np.load(os.path.join(path.path_labels, f)))
        #     subj_end_idx = current_idx + num_labels
        #     # Check for overlap
        #     if any(current_idx <= test_idx < subj_end_idx for test_idx in test_indices_for_fold):
        #         test_subjects.add(subj_id_from_file)
        #     current_idx = subj_end_idx
        # print(f"Subjects in Fold {fold_id} test set: {sorted(list(test_subjects))}")
        return
    else:
        print(f"Subject {subject_id} IS in the test set for Fold {fold_id}.")
        # Map global indices to local indices within the test set
        subject_local_indices = [np.where(test_indices_for_fold == idx)[0][0] for idx in subject_indices_in_fold_test_set]
        print(f"Subject global indices range: [{start_idx_global}, {end_idx_global}) -> Local indices in test set: {len(subject_local_indices)} epochs")

    # 5. Load the model for the target fold
    model_path = os.path.join(f'./Kfold_models/fold{fold_id}/model.pkl')
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    print(f"Loading model from: {model_path}")
    model = Transformer(config)
    model = model.to(config.device)
    model.load_state_dict(torch.load(model_path, map_location=config.device), strict=True)
    model.eval()

    # 6. Extract the subject's data *from the test set*
    X_test_fold = dataset[test_indices_for_fold]
    y_test_fold = labels[test_indices_for_fold]
    
    X_subject = X_test_fold[subject_local_indices]
    y_subject_true = y_test_fold[subject_local_indices]
    
    # Verify against originally loaded labels
    if not np.array_equal(y_subject_true.numpy(), subject_true_labels):
         print("Warning: Extracted labels do not match originally loaded subject labels. Check indexing.")
         print(f"Extracted shape: {y_subject_true.shape}, Original shape: {subject_true_labels.shape}")
         # Fallback to originally loaded labels if lengths match
         if len(y_subject_true) == len(subject_true_labels):
             y_subject_true = torch.from_numpy(subject_true_labels).long()
             print("Using originally loaded labels.")
         else:
             print("Label length mismatch. Cannot proceed.")
             return

    print(f"Subject data shape: {X_subject.shape}, Subject labels shape: {y_subject_true.shape}")

    # 7. Perform prediction
    subject_dataset = TensorDataset(X_subject, y_subject_true)
    subject_loader = DataLoader(dataset=subject_dataset, batch_size=config.batch_size, shuffle=False)
    
    all_preds = []
    print("Running prediction on subject data...")
    with torch.no_grad():
        for data, _ in tqdm(subject_loader):
            data = data.to(config.device)
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
    
    y_subject_pred = np.array(all_preds)
    y_subject_true_np = y_subject_true.cpu().numpy()

    # 8. Calculate metrics
    accuracy = accuracy_score(y_subject_true_np, y_subject_pred)
    f1 = f1_score(y_subject_true_np, y_subject_pred, average='macro')
    print(f"Subject {subject_id} - Fold {fold_id} Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro F1-Score: {f1:.4f}")

    # 9. Plot hypnogram
    save_filename = f"hypnogram_subject_{subject_id}_fold_{fold_id}.png"
    save_filepath = os.path.join(path.path_figure, save_filename)
    plot_hypnogram(y_subject_true_np, y_subject_pred, subject_id, fold_id, accuracy, f1, save_filepath)

if __name__ == '__main__':
    config = Config()
    path = Path()

    TARGET_SUBJECT_ID = 'SC4071E0' # Example subject ID from user request
    TARGET_FOLD_ID = 2 # Target fold from user request

    # Ensure figure directory exists
    if not os.path.exists(path.path_figure):
        os.makedirs(path.path_figure)
        print(f"Created figure directory: {path.path_figure}")

    main(config, path, TARGET_SUBJECT_ID, TARGET_FOLD_ID)