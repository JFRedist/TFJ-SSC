import torch
from torch import nn

from model import PositionalEncoding


class SeqTFJSSC(nn.Module):
    """Slim TFJ-SSC epoch encoder followed by a Transformer across consecutive epochs.

    Input:  (batch, L, 3, 29, 128) — L consecutive 30-s epochs, 3 channels of TF images
    Output: (batch, L, num_classes) — one prediction per epoch
    """

    def __init__(self, config):
        super().__init__()
        d = config.dim_model
        self.num_channels = 3

        # single-channel feature extraction: one encoder shared by all channels, told apart by a channel embedding
        self.position_single = PositionalEncoding(d_model=d, dropout=0.1)
        self.channel_embedding = nn.Parameter(torch.zeros(self.num_channels, 1, d))
        nn.init.normal_(self.channel_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=config.num_head, dim_feedforward=config.slim_forward_hidden,
                                           dropout=config.dropout, batch_first=True)
        self.encoder_single = nn.TransformerEncoder(layer, num_layers=config.slim_num_encoder)

        # multi-channel feature fusion with a residual connection, as in TFJ-SSC
        self.drop = nn.Dropout(p=0.5)
        self.layer_norm = nn.LayerNorm(d * self.num_channels)
        self.position_multi = PositionalEncoding(d_model=d * self.num_channels, dropout=0.1)
        layer_multi = nn.TransformerEncoderLayer(d_model=d * self.num_channels, nhead=config.num_head,
                                                 dim_feedforward=config.slim_forward_hidden * 2,
                                                 dropout=config.dropout, batch_first=True)
        self.encoder_multi = nn.TransformerEncoder(layer_multi, num_layers=config.slim_num_encoder_multi)

        # attention pooling over the 29 time frames instead of flatten + large FC layer
        self.pool_score = nn.Linear(d * self.num_channels, 1)
        self.project = nn.Sequential(nn.Linear(d * self.num_channels, config.seq_dim), nn.GELU(), nn.Dropout(0.1))

        # sequence encoder across epochs
        self.position_seq = PositionalEncoding(d_model=config.seq_dim, dropout=0.1, max_len=128)
        layer_seq = nn.TransformerEncoderLayer(d_model=config.seq_dim, nhead=config.num_head,
                                               dim_feedforward=config.seq_dim * 4, dropout=config.dropout,
                                               batch_first=True)
        self.encoder_seq = nn.TransformerEncoder(layer_seq, num_layers=config.seq_num_encoder)
        self.classifier = nn.Linear(config.seq_dim, config.num_classes)

    def encode_epochs(self, x):
        """x: (N, 3, 29, 128) -> (N, seq_dim)"""
        n, c, t, f = x.shape
        x = self.position_single(x.reshape(n * c, t, f))
        x = x.view(n, c, t, f) + self.channel_embedding        # (N, 3, 29, 128)
        x = self.encoder_single(x.reshape(n * c, t, f))
        x = x.view(n, c, t, f).permute(0, 2, 1, 3).reshape(n, t, c * f)  # concat channels: (N, 29, 384)

        x = self.layer_norm(self.drop(x))
        residual = x
        x = self.encoder_multi(self.position_multi(x))
        x = self.layer_norm(x + residual)                        # residual connection

        weights = torch.softmax(self.pool_score(x), dim=1)       # (N, 29, 1)
        x = (weights * x).sum(dim=1)                             # (N, 384)
        return self.project(x)

    def forward(self, x):
        b, l = x.shape[:2]
        e = self.encode_epochs(x.reshape(b * l, *x.shape[2:])).view(b, l, -1)
        e = self.encoder_seq(self.position_seq(e))
        return self.classifier(e)
