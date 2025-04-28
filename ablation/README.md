# Channel ablation variants

Drop-in replacements for `model.py` and `data_loader.py` in the project root, used for the channel ablation study (thesis Table 5-4).

| Variant | Input channels | Acc (%) | κ | MF1 (%) |
|---|---|---|---|---|
| `single_eeg/` | EEG Fpz-Cz | 83.98 | 0.78 | 77.34 |
| `no_eog/` | EEG Fpz-Cz + EEG Pz-Oz | 86.18 | 0.81 | 80.98 |
| root (full model) | EEG Fpz-Cz + EEG Pz-Oz + EOG | 87.46 | 0.83 | 83.05 |

To run a variant, copy its two files over the ones in the project root and run `Kfold_trainer.py` as usual.
