from __future__ import annotations

import copy
import torch


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = float(decay); self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters(): parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema, current in zip(self.model.parameters(), model.parameters()):
            ema.mul_(self.decay).add_(current, alpha=1 - self.decay)
        for ema, current in zip(self.model.buffers(), model.buffers()): ema.copy_(current)

    def state_dict(self): return self.model.state_dict()
    def load_state_dict(self, state): self.model.load_state_dict(state)

