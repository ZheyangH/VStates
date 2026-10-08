"""Object-track-local, ordered sliding-window state-transition encoding."""
import torch
from torch import nn


class StateTransitionRepresentation(nn.Module):
    """Encode valid windows; fuse their summary with the original OME.

    Inputs: states [N, T, D], valid [N, T]. No semantic state or role
    labels are assumed. Attention is local to each object's window.
    """

    def __init__(self, state_dim, window_size=3, stride=1, hidden_dim=128,
                 mode="mlp", heads=4, dropout=0.0):
        super().__init__()
        if window_size < 2 or stride < 1 or hidden_dim < 1:
            raise ValueError("window_size >= 2, stride >= 1, hidden_dim >= 1 required")
        if mode not in ("mlp", "attention"):
            raise ValueError("mode must be mlp or attention")
        if mode == "attention" and (heads < 1 or hidden_dim % heads):
            raise ValueError("attention hidden_dim must be divisible by heads")
        self.state_dim = state_dim
        self.window_size = window_size
        self.stride = stride
        self.mode = mode
        # Compress high-dimensional SlowFast features before window encoding.
        self.project = nn.Linear(state_dim, hidden_dim)
        if mode == "mlp":
            self.encoder = nn.Sequential(
                nn.Linear((2 * window_size - 1) * hidden_dim, hidden_dim),
                nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim))
        else:
            self.position = nn.Parameter(torch.randn(window_size, hidden_dim) * 0.02)
            self.delta_project = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.encoder = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=heads, dim_feedforward=2 * hidden_dim,
                dropout=dropout, activation="gelu")
        self.output = nn.Linear(hidden_dim, state_dim, bias=False)
        # Small initial residual, so the pretrained OME remains useful.
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def transition_summary(self, states, valid):
        if states.ndim != 3 or states.shape[-1] != self.state_dim:
            raise ValueError("states must have shape [N, T, state_dim]")
        if valid.shape != states.shape[:2]:
            raise ValueError("valid must have shape [N, T]")
        valid = valid.to(device=states.device, dtype=torch.bool)
        if states.shape[1] < self.window_size:
            return states.new_zeros(states.shape[0], self.state_dim)
        # Missing observations never form transitions, including across gaps.
        clean = torch.where(valid.unsqueeze(-1), states, torch.zeros_like(states))
        projected = self.project(clean)
        windows = projected.unfold(1, self.window_size, self.stride).permute(0, 1, 3, 2)
        window_valid = valid.unfold(1, self.window_size, self.stride).all(dim=-1)
        delta = windows[:, :, 1:] - windows[:, :, :-1]
        if self.mode == "mlp":
            encoded = self.encoder(torch.cat(
                [windows.flatten(2), delta.flatten(2)], dim=-1))
        else:
            padded_delta = torch.cat([torch.zeros_like(windows[:, :, :1]), delta], dim=2)
            tokens = windows + self.delta_project(padded_delta) + self.position
            n, count, width, dim = tokens.shape
            encoded = self.encoder(tokens.reshape(n * count, width, dim).transpose(0, 1))
            encoded = encoded.mean(dim=0).reshape(n, count, dim)
        encoded = torch.where(window_valid.unsqueeze(-1), encoded, torch.zeros_like(encoded))
        summary = encoded.sum(dim=1) / window_valid.sum(dim=1, keepdim=True).clamp_min(1)
        return self.output(summary)

    def forward(self, states, valid, baseline):
        return baseline + torch.sigmoid(self.gate) * self.transition_summary(states, valid)
