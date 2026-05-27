"""
1D CNN for Human Activity Recognition — nRF52840 TinyML deployment.

Architecture targets
--------------------
* ~95 K trainable parameters
* ~95 KB int8 TFLite model   (fits in 1 MB Flash)
* ~48 KB tensor arena         (fits in 256 KB RAM alongside program)
* > 93% accuracy on UCI HAR test split

Input
-----
  (batch, n_channels=6, seq_len=128)    — channels-first, PyTorch Conv1d
  Channel order: body_acc_x/y/z, body_gyro_x/y/z

Output
------
  (batch, n_classes=6) — raw logits; apply softmax for probabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ─────────────────────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    """Conv1d -> BatchNorm1d -> ReLU, with optional MaxPool1d(2) afterwards."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        padding: int,
        pool: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.bn   = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(2) if pool else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        if self.pool is not None:
            x = self.pool(x)
        return x


# ── Main model ──────────────────────────────────────────────────────────────────

class HARCNN(nn.Module):
    """
    Four-block 1D CNN for UCI HAR 6-class activity recognition.

    Architecture
    ------------
    Block 1: Conv1d(6->32,  k=5, p=2) + BN + ReLU             (n, 32, 128)
    Block 2: Conv1d(32->64, k=5, p=2) + BN + ReLU + Pool/2    (n, 64,  64)
    Block 3: Conv1d(64->128,k=3, p=1) + BN + ReLU + Pool/2    (n,128,  32)
    Block 4: Conv1d(128->128,k=3,p=1) + BN + ReLU + Pool/2    (n,128,  16)
    GlobalAvgPool                                               (n, 128)
    FC(128->64)  + ReLU + Dropout(p)                           (n,  64)
    FC(64->6)                                                   (n,   6)

    Approximate parameter counts
    ----------------------------
    block1 conv: 6*32*5 = 960 (+32 BN)
    block2 conv: 32*64*5 = 10240 (+64 BN)
    block3 conv: 64*128*3 = 24576 (+128 BN)
    block4 conv: 128*128*3 = 49152 (+128 BN)
    fc1: 128*64 = 8192 (+64)
    fc2: 64*6 = 384 (+6)
    Total ≈ 95 K
    """

    def __init__(
        self,
        n_channels: int = 6,
        n_classes: int = 6,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes

        # Feature extractor
        self.block1 = ConvBnRelu(n_channels, 32,  kernel_size=5, padding=2, pool=False)
        self.block2 = ConvBnRelu(32,         64,  kernel_size=5, padding=2, pool=True)
        self.block3 = ConvBnRelu(64,         128, kernel_size=3, padding=1, pool=True)
        self.block4 = ConvBnRelu(128,        128, kernel_size=3, padding=1, pool=True)

        # Aggregation
        self.gap = nn.AdaptiveAvgPool1d(1)   # -> (n, 128, 1)

        # Classifier head
        self.fc1     = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2     = nn.Linear(64, n_classes)

        # Weight initialisation
        self._init_weights()

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (batch, n_channels, seq_len)

        Returns
        -------
        logits : Tensor (batch, n_classes)
        """
        x = self.block1(x)            # (n, 32, 128)
        x = self.block2(x)            # (n, 64,  64)
        x = self.block3(x)            # (n,128,  32)
        x = self.block4(x)            # (n,128,  16)
        x = self.gap(x).squeeze(-1)   # (n, 128)
        x = F.relu(self.fc1(x))       # (n, 64)
        x = self.dropout(x)
        x = self.fc2(x)               # (n, 6)
        return x

    # ── convenience ──────────────────────────────────────────────────────────

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax probabilities (no gradient)."""
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=-1)

    @property
    def num_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── init ─────────────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


# ── Smoke test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = HARCNN()
    print(model)
    print(f"\nTotal parameters: {model.num_parameters:,}")
    x = torch.randn(4, 6, 128)
    logits = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {logits.shape}")   # (4, 6)
    proba  = model.predict_proba(x)
    print(f"Probabilities sum: {proba.sum(dim=-1)}")  # should be [1, 1, 1, 1]
