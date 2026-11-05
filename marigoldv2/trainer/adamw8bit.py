import bitsandbytes as bnb
import torch


class AdamW8bit(bnb.optim.AdamW8bit):
    """bitsandbytes' AdamW8bit without its torch.cuda.synchronize() after every
    parameter's update (bitsandbytes 0.49 does it unconditionally; only paged states
    need it): the updates are stream-ordered like every other kernel, so the results
    are the same, and the GPU no longer idles between ~600 small updates per step."""

    @torch.no_grad()
    def step(self, closure=None):
        if self.is_paged:
            return super().step(closure)
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if not self.initialized:
            self.check_overrides()
            self.to_gpu()
            self.initialized = True
        for gindex, group in enumerate(self.param_groups):
            for pindex, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                if len(self.state[p]) == 0:
                    self.init_state(group, p, gindex, pindex)
                self.update_step(group, p, gindex, pindex)
        return loss
