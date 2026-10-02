import torch
import os


class Config(object):
    """args in model and trainer"""
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_fold = 5
        self.num_classes = 5
        self.num_epochs = 200               # Because early stopping is used, this parameter can be relatively large
        self.batch_size = 448
        self.pad_size = 29                  # time dimension of TF image
        self.learning_rate = 2e-5
        self.dropout = 0.1                  # dropout rate in transformer encoder
        self.dim_model = 128                # frequency of TF image
        self.forward_hidden = 1024          # hidden units of transformer encoder
        self.fc_hidden = 1024               # hidden units of FC layers
        self.num_head = 8                   # number of heads in transformer encoder
        self.num_encoder = 12               # number of encoders in single-channel feature extraction block
        self.num_encoder_multi = 5          # number of encoders in multi-channel feature fusion block
        self.use_relative_pos = False        # whether to use relative positional encoding
        self.max_relative_position = 20     # maximum relative position for relative positional encoding
        self.lr_scheduler_t_max = 200      # 余弦退火周期长度，通常设置为总epoch数
        self.lr_scheduler_eta_min = 1e-7   # 学习率下限
        self.dynamic_lr_schedule = True    # 是否使用动态学习率调度（根据早停指标调整）
        self.lr_patience_ratio = 2.0       # 动态学习率周期与早停patience的比例
        # Evaluation protocol:
        #   'subject': folds, and the validation set inside each fold, never share a subject (comparable with the literature)
        #   'epoch':   30-s epochs are split at random, as in the thesis (subjects leak across train/val/test)
        self.early_stop_metric = 'f1'       # 'f1' (validation macro-F1, thesis) or 'acc' (validation accuracy, upstream)
        self.early_stop_patience = 10
        self.cv_mode = 'subject'
        self.val_ratio = 0.1                # subject mode: fraction of training subjects held out for validation
        self.seed = 0
        self.use_amp = True                 # bf16 autocast on CUDA


class SeqConfig(Config):
    """Slim epoch encoder + inter-epoch sequence Transformer (seq_trainer.py)"""
    def __init__(self):
        super().__init__()
        self.seq_len = 15                   # epochs per input sequence (1 = no inter-epoch context)
        self.epochs_per_batch = 480         # sequences per batch = epochs_per_batch // seq_len (32 for L = 15)
        self.seq_eval_stride = 5            # window stride at evaluation (capped at seq_len); overlapping predictions are averaged
        self.slim_num_encoder = 2           # encoder layers per channel (shared across channels)
        self.slim_num_encoder_multi = 2     # encoder layers in the multi-channel fusion block
        self.slim_forward_hidden = 512      # FFN size of the per-channel encoder (fusion block uses 2x)
        self.seq_dim = 128                  # epoch embedding size fed to the sequence encoder
        self.seq_num_encoder = 2            # encoder layers across epochs
        self.learning_rate = 3e-4           # chosen on validation MF1 (3e-4 to 5e-4 equally good; runs A/B/C used 1e-4)
        self.class_weighting = 'none'       # 'none' or 'inv_sqrt' (weights ~ 1/sqrt(class frequency))
        # learning-rate schedule:
        #   'restart' (thesis): cosine down to eta_min and back up, driven by the early-stopping counter
        #   'plateau': multiply by plateau_factor after every plateau_patience epochs without improvement, never back up
        #   'cosine':  cosine decay to eta_min over cosine_epochs, then stop (early stopping only selects the checkpoint)
        self.lr_schedule = 'restart'
        self.warmup_epochs = 0              # linear warmup per step over the first warmup_epochs epochs
        self.plateau_factor = 0.3
        self.plateau_patience = 3
        self.cosine_epochs = 40


class Path(object):
    """path of files in this project"""
    def __init__(self):
        self.path_PSG = 'dataset/sleepEDF-78/sleep-cassette'
        self.path_hypnogram = 'dataset/sleepEDF-78/Hypnogram'
        self.path_raw_data = 'data/sleepEDF-78/data_array/raw_data'
        self.path_labels = 'data/sleepEDF-78/data_array/raw_data/labels'
        self.path_TF = 'data/sleepEDF-78/data_array/TF_data'
        self.path_figure = 'figures'

        if not os.path.exists(self.path_hypnogram):
            os.makedirs(self.path_hypnogram)

        if not os.path.exists(self.path_raw_data):
            os.makedirs(self.path_raw_data)

        if not os.path.exists(self.path_TF):
            os.makedirs(self.path_TF)

        if not os.path.exists(self.path_figure):
            os.makedirs(self.path_figure)
