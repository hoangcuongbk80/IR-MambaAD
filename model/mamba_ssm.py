"""

- BaseSSMCell stores parameter matrices A,B,C,D.
- SelectiveSSM computes time-varying parameter matrices as convex combinations of K base tuples
  using provided alpha weights (alpha shape: (B,T,K)), and runs the linear recurrence.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class BaseSSMCell(nn.Module):
    def __init__(self, input_dim: int, state_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        # A: (state_dim, state_dim), B: (state_dim, input_dim)
        # C: (input_dim, state_dim), D: (input_dim, input_dim) -> output dim == input_dim
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(state_dim, input_dim) * 0.01)
        self.C = nn.Parameter(torch.randn(input_dim, state_dim) * 0.01)
        self.D = nn.Parameter(torch.randn(input_dim, input_dim) * 0.01)

    def forward(self, x_t, h_prev):
        # x_t: (B, input_dim), h_prev: (B, state_dim)
        h_next = torch.matmul(h_prev, self.A.t()) + torch.matmul(x_t, self.B.t())
        y_t = torch.matmul(h_next, self.C.t()) + torch.matmul(x_t, self.D.t())
        return y_t, h_next


class SelectiveSSM(nn.Module):
    def __init__(self, input_dim: int, state_dim: int, K: int = 4):
        super().__init__()
        self.K = K
        self.input_dim = input_dim
        self.state_dim = state_dim
        # create K base cells
        self.cells = nn.ModuleList([BaseSSMCell(input_dim=input_dim, state_dim=state_dim) for _ in range(K)])

    def forward(self, x_seq: torch.Tensor, alpha_seq: torch.Tensor):
        """
        x_seq: (B, T, D) where D == input_dim
        alpha_seq: (B, T, K) convex weights per time-step (softmax assumed)
        returns:
            y_seq: (B, T, D)
            h_T: (B, state_dim)
        Naive implementation: compute combined params per time-step by weighted sum and apply recurrence.
        """
        B, T, D = x_seq.shape
        device = x_seq.device
        h = torch.zeros(B, self.state_dim, device=device, dtype=x_seq.dtype)

        # pre-extract base params as tensors for faster weighted sum
        As = torch.stack([c.A for c in self.cells], dim=0)  # (K, S, S)
        Bs = torch.stack([c.B for c in self.cells], dim=0)  # (K, S, D)
        Cs = torch.stack([c.C for c in self.cells], dim=0)  # (K, D, S)
        Ds = torch.stack([c.D for c in self.cells], dim=0)  # (K, D, D)

        ys = []
        for t in range(T):
            x_t = x_seq[:, t, :]          # (B, D)
            alpha_t = alpha_seq[:, t, :]  # (B, K)
            # compute weighted parameters for each batch separately
            # Expand alpha to (B, K, 1, 1) and sum
            alpha_t_exp = alpha_t.view(B, self.K, 1, 1)
            # weighted sum of As -> (B, S, S)
            A_t = (alpha_t_exp * As.unsqueeze(0)).sum(dim=1)
            B_t = (alpha_t_exp * Bs.unsqueeze(0)).sum(dim=1)
            # Cs and Ds shapes: (K, D, S) and (K, D, D)
            C_t = (alpha_t_exp * Cs.unsqueeze(0)).sum(dim=1)
            D_t = (alpha_t_exp * Ds.unsqueeze(0)).sum(dim=1)

            # compute h_next: h_prev @ A_t^T + x_t @ B_t^T
            # h: (B, S), A_t: (B, S, S) -> perform batched matmul
            hA = torch.bmm(h.unsqueeze(1), A_t).squeeze(1)  # (B, S)
            xB = torch.bmm(x_t.unsqueeze(1), B_t.transpose(1,2)).squeeze(1)  # (B, S)
            h = hA + xB
            # output y = h @ C_t^T + x @ D_t^T
            y_h = torch.bmm(h.unsqueeze(1), C_t.transpose(1,2)).squeeze(1)  # (B, D)
            y_x = torch.bmm(x_t.unsqueeze(1), D_t.transpose(1,2)).squeeze(1)
            y = y_h + y_x
            ys.append(y.unsqueeze(1))
        y_seq = torch.cat(ys, dim=1)
        return y_seq, h
