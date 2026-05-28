import torch
import torch.optim
import logging


class SODAWrapper(torch.optim.Optimizer):
    r"""
    OBS: Do not use weight decay for the underlying optimizer.
    """
    def __init__(self, 
                 base : torch.optim.Optimizer,
                 **kwargs,
                 ):
        
        # Loop over the keys in the kwargs dictionary
        for key in kwargs:
            logging.warning(
                f"Unrecognized optimizer argument '{key}'. It will be ignored."
            )

        self.base = base
        for group in self.base.param_groups:
            if 'k' not in group:
                group['k'] = 0

    def add_param_group(self, param_group):
        if 'k' not in param_group:
            param_group['k'] = 0
        return self.base.add_param_group(param_group)

    def load_state_dict(self, state_dict):
        self.base.load_state_dict(state_dict)
        for group in self.base.param_groups:
            if 'k' not in group:
                group['k'] = 0

    def state_dict(self):
        return self.base.state_dict()
        
    def zero_grad(self, set_to_none=True):
        return self.base.zero_grad(set_to_none)

    @property
    def param_groups(self):
        return self.base.param_groups

    @property
    def state(self):
        return self.base.state

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Keep a local snapshot before base.step(); do not mutate optimizer state yet.
        prev_map = {}
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                prev_map[p] = torch.clone(p, memory_format=torch.preserve_format)

        self.base.step()

        for group in self.param_groups:
            k = group['k']
            for p in group['params']:
                if p not in prev_map:
                    continue
                state = self.state[p]
                prev = prev_map[p]

                if 'z0' not in state:
                    state['z0'] = torch.clone(prev, memory_format=torch.preserve_format)

                z = state['z0']
                z = z.add((k+2) * (p - prev))
                
                x = prev * (1 - 1/(k+2)) + z * (1/(k+2))
                p.copy_(x)
            
            group['k'] = k+1

        return loss
