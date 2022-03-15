from copy import deepcopy
from torch.nn import Module


class EMAWrapper(Module):
    def __init__(self, net, decay=0.98):
        super().__init__()
        self.net = net
        self.net_ema = deepcopy(self.net)
        self.net_ema.requires_grad_(False)
        self.decay = decay

    def forward(self, *args, **kwargs):
        if self.training:
            return self.net(*args, **kwargs)
        else:
            return self.net_ema(*args, **kwargs)

    def step(self, decay=None):
        if decay is None:
            decay = self.decay
        for (k, v), (k_ema, v_ema) in zip(self.net.named_parameters(), self.net_ema.named_parameters()):
            assert k == k_ema
            v_ema.mul_(decay).add_(v, alpha=1-decay)
