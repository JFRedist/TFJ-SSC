# TFJ-SSC

**Temporal-Frequency Joint Transformer for Sleep Stage Classification**
基于 Transformer 的脑电-眼电（EEG-EOG）多模态时频联合睡眠分期模型

Code for my undergraduate thesis *脑电-眼电信号融合的多模态睡眠阶段监测系统* (EEG-EOG Fusion-Based Multimodal Sleep Stage Monitoring System), School of Artificial Intelligence, Tiangong University, 2025.

This repository is a fork of [MultiChannelSleepNet](https://github.com/yangdai97/MultiChannelSleepNet) (Dai et al., IEEE JBHI 2023). TFJ-SSC keeps its overall design (per-channel Transformer feature extraction, then multi-channel fusion) and changes the training procedure and tooling; see [What changed](#what-changed-compared-with-multichannelsleepnet).

![TFJ-SSC architecture](docs/figures/architecture.png)

## Method

Each 30 s epoch of three PSG channels (EEG Fpz-Cz, EEG Pz-Oz, horizontal EOG) is turned into a time-frequency image with the STFT and z-score normalized per channel. TFJ-SSC then:

1. **Single-channel feature extraction**: adds a sinusoidal positional encoding to each channel's image and passes it through its own 12-layer Transformer encoder.
2. **Multi-channel feature fusion**: concatenates the three feature maps, applies layer normalization and positional encoding, and runs a 5-layer Transformer encoder. A residual connection keeps each channel's original features.
3. **Classifier**: two fully connected layers and softmax over the five AASM stages (W, N1, N2, N3, REM).

Training uses 5-fold stratified cross-validation, AdamW (weight decay 0.01), early stopping on validation macro-F1 (patience 10), and a cosine-like learning-rate schedule driven by the early-stopping counter (2e-5 → 1e-7).

## Results (Sleep-EDF-78, 5-fold CV)

| Protocol | Accuracy | Cohen's κ | Macro-F1 | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|---|
| Subject-wise (`cv_mode = 'subject'`) | 78.81 ± 0.81 % | 0.710 ± 0.010 | 72.50 ± 0.99 % | 91.37 | 39.53 | 81.75 | 76.97 | 72.78 |
| Epoch-wise (`cv_mode = 'epoch'`, thesis) | 87.46 % | 0.827 | 83.05 % | 95.42 | 59.32 | 88.36 | 85.41 | 86.76 |

Overall metrics are averaged over folds (± standard deviation across the 5 folds); per-class columns are F1 scores (%) from the pooled confusion matrix. The subject-wise baseline was trained on an RTX 5070 with the thesis hyperparameters (66 minutes for all 5 folds; early stopping after 17–23 epochs per fold). Channel ablation results are in [`ablation/`](ablation/README.md).

**Evaluation protocol.** The thesis results use *epoch-wise* cross-validation: 30-s epochs are split at random, so epochs of the same subject appear in both training and test sets. This overestimates performance by about 9 accuracy points and is not comparable with published results, which use subject-wise splits. The default is now `cv_mode = 'subject'`, where test folds and validation sets never share a subject with the training set (each fold: about 56 training, 7 validation and 15 test subjects).

![Training and validation accuracy, fold 2](docs/figures/training_fold2.png)

The trained weights were not kept, so none are included. To reproduce the results, train the model with the steps below.

## Requirements

Python 3.10 or 3.11 and PyTorch 2.x with CUDA. The thesis results were trained on one RTX 4090 with batch size 448. With bf16 autocast (`use_amp = True`, the default), batch size 448 needs about 5 GB of GPU memory; on an RTX 5070 one training epoch takes about 40 s. RTX 50-series GPUs need a PyTorch build for CUDA 12.8 or newer:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

```bash
pip install -r requirements.txt
```

## Usage

1. Download the `sleep-cassette` recordings of [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/) (both `*-PSG.edf` and `*-Hypnogram.edf`) into `dataset/sleepEDF-78/sleep-cassette/`.
2. `python dataset_prepare.py` moves the hypnograms to `dataset/sleepEDF-78/Hypnogram/` and extracts 30 s epochs of the three channels.
3. `python data_preprocess_TF.py` computes the STFT images and normalizes each channel.
4. `python Kfold_trainer.py` runs 5-fold cross-validation and saves models, curves and the fold split to `Kfold_models/fold*/`.
5. `python result_evaluate.py` prints the confusion matrix and metrics.

Both scripts accept `--cv-mode subject|epoch` and `--folds 0 1 ...`; the trainer also accepts `--epochs` and `--batch-size`. A quick check that everything runs: `python Kfold_trainer.py --folds 0 --epochs 1`.

Optional:

- `python visualize_training.py` plots loss and accuracy curves per fold.
- `python visualize_subject_results.py` plots the true and predicted hypnogram of one subject.
- `python generate_stft_example.py` plots an STFT example.

Hyperparameters are in `args.py`. Set `use_relative_pos = True` there to try relative positional encoding instead of the default absolute encoding.

## What changed compared with MultiChannelSleepNet

- Ported to PyTorch 2.x (`batch_first` encoders, no `autograd.Variable`)
- Macro-F1 is tracked during training and used as the early-stopping criterion
- Learning-rate schedule tied to the early-stopping counter
- Optional relative positional encoding (`relative_position.py`, `relative_transformer.py`)
- Deterministic file order in preprocessing and data loading
- Subject-wise cross-validation with subject-disjoint validation sets (`cv_mode`)
- Training data reshuffled every epoch (upstream iterated in fixed recording order), bf16 autocast, float32 TF storage
- Channel ablation variants (`ablation/`) and visualization scripts
- Early CNN baseline experiments on raw signals (`experiments/early_simplesleepnet/`)

## Repository layout

```
├── args.py                    hyperparameters and paths
├── dataset_prepare.py         EDF → 30 s epochs
├── data_preprocess_TF.py      epochs → normalized STFT images
├── data_loader.py
├── model.py                   TFJ-SSC
├── relative_position.py       relative positional encoding (optional)
├── relative_transformer.py
├── early_stop_tool.py
├── Kfold_trainer.py           5-fold training
├── result_evaluate.py
├── visualize_*.py, generate_stft_example.py
├── ablation/                  no-EOG and single-EEG variants
├── experiments/               early baseline experiments
└── docs/figures/
```

## Acknowledgements

This project builds on the code of MultiChannelSleepNet. If you use it, please also cite the original paper:

```bibtex
@article{dai2023multichannelsleepnet,
  title   = {MultiChannelSleepNet: A Transformer-Based Model for Automatic Sleep Stage Classification With PSG},
  author  = {Dai, Yang and Li, Xiuli and Liang, Shanshan and Wang, Lukang and Duan, Qingtian and Yang, Hui and Zhang, Chunqing and Chen, Xiaowei and Li, Longhui and Li, Xingyi and Liao, Xiang},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  volume  = {27},
  number  = {9},
  pages   = {4204--4215},
  year    = {2023},
  doi     = {10.1109/JBHI.2023.3284160}
}
```

The upstream repository does not declare a license, so none is added here. Code taken from MultiChannelSleepNet remains the work of its original authors.

Data: [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/), PhysioNet.
