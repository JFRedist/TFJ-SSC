import os
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedGroupKFold, GroupShuffleSplit

from args import Config, Path


def load_dataset(path_labels, path_dataset, return_records=False):
    """Load all TF images and labels.

    Returns:
        dataset (Tensor float32): N x 3 x 29 x 128 (EEG Fpz-Cz, EEG Pz-Oz, EOG)
        labels (Tensor int64): N
        subjects (ndarray): subject ID of every epoch, parsed from the record name SC4ssN.. (ss = subject, N = night)
        records (ndarray, only if return_records): index of the recording (night) of every epoch; epochs of one
            recording are contiguous and in temporal order
    """
    dir_annotation = sorted(os.listdir(path_labels))

    labels, subjects, records = [], [], []
    for r, f in enumerate(dir_annotation):
        y = np.load(os.path.join(path_labels, f))
        labels.append(y)
        subjects.append(np.full(len(y), int(f[3:5])))
        records.append(np.full(len(y), r))
    labels = torch.from_numpy(np.concatenate(labels)).long()
    subjects = np.concatenate(subjects)
    records = np.concatenate(records)

    channels = []
    for name in ['TF_EEG_Fpz-Cz_mean_std.npy', 'TF_EEG_Pz-Oz_mean_std.npy', 'TF_EOG_mean_std.npy']:
        channels.append(np.load(os.path.join(path_dataset, name)).astype('float32', copy=False))
    dataset = torch.from_numpy(np.stack(channels, axis=1))
    del channels

    assert len(dataset) == len(labels), 'TF data and labels are misaligned'
    print('dataset:', tuple(dataset.shape), '| subjects:', len(np.unique(subjects)), '| records:', len(dir_annotation))
    if return_records:
        return dataset, labels, subjects, records
    return dataset, labels, subjects


def get_folds(labels, subjects, config):
    """Deterministic fold definitions: a list of dicts with 'train', 'val' and 'test' index arrays."""
    y = labels.numpy()
    idx = np.arange(len(y))
    folds = []

    if config.cv_mode == 'epoch':
        # Thesis setting: one validation set of 1/(k+1) of all epochs, then stratified k-fold on the rest
        idx_tt, idx_val = train_test_split(idx, test_size=1 / (config.num_fold + 1), random_state=0, stratify=y)
        kf = StratifiedKFold(n_splits=config.num_fold, shuffle=True, random_state=0)
        for tr, te in kf.split(idx_tt, y[idx_tt]):
            folds.append({'train': idx_tt[tr], 'val': idx_val, 'test': idx_tt[te]})

    elif config.cv_mode == 'subject':
        # Test subjects per fold; validation subjects are drawn from the remaining training subjects
        kf = StratifiedGroupKFold(n_splits=config.num_fold, shuffle=True, random_state=config.seed)
        for fold, (tr_all, te) in enumerate(kf.split(idx, y, subjects)):
            gss = GroupShuffleSplit(n_splits=1, test_size=config.val_ratio, random_state=config.seed + fold)
            tr, va = next(gss.split(tr_all, y[tr_all], subjects[tr_all]))
            folds.append({'train': tr_all[tr], 'val': tr_all[va], 'test': te})
            assert not set(subjects[tr_all[tr]]) & set(subjects[te]), 'subject leak between train and test'
            assert not set(subjects[tr_all[tr]]) & set(subjects[tr_all[va]]), 'subject leak between train and val'

    else:
        raise ValueError(f'unknown cv_mode: {config.cv_mode}')

    return folds


def iterate_batches(dataset, labels, idx, batch_size, shuffle=False, generator=None):
    """Yield (data, target) batches by fancy indexing, without copying the whole split into a new tensor."""
    if shuffle:
        idx = idx[torch.randperm(len(idx), generator=generator).numpy()]
    for start in range(0, len(idx), batch_size):
        b = torch.from_numpy(idx[start:start + batch_size])
        yield dataset[b], labels[b]


def record_ranges(idx, records):
    """[start, end) global index ranges of every recording contained in a split (splits hold whole recordings)."""
    idx = np.sort(idx)
    cuts = np.flatnonzero(np.diff(records[idx]) != 0) + 1
    segments = np.split(idx, cuts)
    assert all(seg[-1] - seg[0] + 1 == len(seg) for seg in segments), 'split contains partial recordings (use cv_mode subject)'
    return [(seg[0], seg[-1] + 1) for seg in segments]


def sequence_starts(ranges, seq_len, stride=None, generator=None):
    """Start indices of length-seq_len windows inside each recording.

    Training (stride=None): non-overlapping windows with a random offset per recording, so every epoch is used
    about once per training epoch and the window borders move between training epochs.
    Evaluation (stride=k): windows every k epochs plus one aligned to the end, so every epoch is covered.
    """
    starts = []
    for s, e in ranges:
        n = e - s
        assert n >= seq_len, 'recording shorter than seq_len'
        if stride is None:
            offset = int(torch.randint(0, seq_len, (1,), generator=generator)) if n > seq_len else 0
            starts.append(np.arange(s + min(offset, n - seq_len), e - seq_len + 1, seq_len))
        else:
            st = np.arange(s, e - seq_len + 1, stride)
            starts.append(np.unique(np.append(st, e - seq_len)))
    return np.concatenate(starts)


def iterate_sequences(dataset, labels, starts, seq_len, batch_size, shuffle=False, generator=None):
    """Yield (data, target, index) batches: data (B, L, 3, 29, 128), target and index (B, L)."""
    if shuffle:
        starts = starts[torch.randperm(len(starts), generator=generator).numpy()]
    offsets = np.arange(seq_len)
    for i in range(0, len(starts), batch_size):
        index = torch.from_numpy(starts[i:i + batch_size, None] + offsets)
        yield dataset[index], labels[index], index


def data_generator(path_labels, path_dataset):
    """Thesis-style loader kept for visualize_subject_results.py (epoch-wise validation hold-out)."""
    config = Config()
    dataset, labels, _ = load_dataset(path_labels, path_dataset)
    X_train_test, X_val, y_train_test, y_val = train_test_split(dataset, labels, test_size=1/(config.num_fold+1), random_state=0, stratify=labels)
    val_loader = DataLoader(dataset=TensorDataset(X_val, y_val), batch_size=config.batch_size, shuffle=False)
    print('val_set:', len(X_val))
    return X_train_test, y_train_test, val_loader
