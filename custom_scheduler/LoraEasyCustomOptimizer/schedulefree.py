# Source: https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/optimizer/sam.py
from typing import Callable, Dict, Optional, Tuple, Union

import torch

from pytorch_optimizer.base.exception import NoClosureError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, OPTIMIZER, PARAMETERS, LOSS
from .utils import copy_stochastic_

class ScheduleFreeWrapper(BaseOptimizer):
    r"""
        Wrap any optimizer to make it Schedule-Free. 
        
        This version uses a memory-efficient swap operation but may be slower than the reference version. In most cases
        the performance difference is negligible.
        For the best possible performance and memory-usage, Schedule-Free needs 
        to be directly integrated with the base optimizer.

        When using this version, you can disable the base optimizer's 
        momentum, as it's no longer necessary when using our wrapper's 
        momentum (although you can use both types of momentum if you want).


        Example usage:
        ```
        base_optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0025)
        optimizer = ScheduleFreeWrapper(
            base_optimizer, momentum=0.9, weight_decay_at_y=0.1)
        ```

        If you set weight decay on the base optimizer, it computes weight decay
        at $z$. We offer the option to compute weight decay at $y$, via the 
        `weight_decay_at_y` parameter, which seems to give better results in 
        our experiments. This approach to decay only works correctly if the base
        optimizer uses group["lr"] as the current learning rate. 

        :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
        base_optimizer (torch.optim.Optimizer): 
            PyTorch optimizer object
        momentum (float): Apply momentum on the outer optimizer (default 0.9)
        weight_decay_at_y (float): 
            Weight decay calculated at the y point. Set weight decay on the 
            inner optimizer to instead calculate at z (default: 0.0).
        r (float): Use polynomial weighting in the average 
            with power r (default 0.0).
        weight_lr_power (float): During warmup, the weights in the average will
            be equal to lr raised to this power. Set to 0 for no weighting
            (default 2.0).
    """
    def __init__(self, 
                 params: PARAMETERS,
                 base_optimizer : OPTIMIZER, 
                 sf_weight_decay_at_y : float = 0.0,
                 sf_momentum : float = 0.9,
                 sf_weight_lr_power : float = 2.0,
                 sf_r : float = 0.0,
                 **kwargs):
        
        self.validate_non_negative(sf_weight_decay_at_y, 'sf_weight_decay_at_y')
        self.validate_non_negative(sf_momentum, 'sf_momentum')
        self.validate_non_negative(sf_weight_lr_power, 'sf_weight_lr_power')
        self.validate_non_negative(sf_r, 'sf_r')

        self.sf_weight_decay_at_y = sf_weight_decay_at_y
        self.sf_weight_lr_power = sf_weight_lr_power
        self.sf_r = sf_r
        self.sf_momentum = sf_momentum
        self.train_mode = False

        defaults: DEFAULTS = {'sf_weight_decay_at_y': sf_weight_decay_at_y, 'sf_momentum': sf_momentum, 'sf_weight_lr_power': sf_weight_lr_power, 'sf_r': sf_r}
        defaults.update(kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    def __str__(self) -> str:
        return 'ScheduleFreeWrapper'

    @torch.no_grad()
    def reset(self):
        pass

    @torch.no_grad()
    def eval(self):
        if self.train_mode:
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p to x
                        p.lerp_(end=state['z'], weight=1-1/self.sf_momentum)
        self.train_mode = False

    @torch.no_grad()
    def train(self):
        if not self.train_mode:
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p to y
                        p.lerp_(end=state['z'], weight=1-self.sf_momentum)
        self.train_mode = True

    @staticmethod
    def swap(x: torch.Tensor, y: torch.Tensor):
        # If this crashes use ScheduleFreeWrapperReference instead
        x.view(torch.uint8).bitwise_xor_(y.view(torch.uint8))
        y.view(torch.uint8).bitwise_xor_(x.view(torch.uint8))
        x.view(torch.uint8).bitwise_xor_(y.view(torch.uint8))

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        if not self.train_mode:
            raise Exception("Optimizer was not in train mode when step is called. "
                            "Please insert .train() and .eval() calls on the "
                            "optimizer. See documentation for details.")
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            if 'k' in group:
                group['k'] += 1
            else:
                group['k'] = 1

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                if 'z' not in state:
                    state['z'] = torch.clone(p, memory_format=torch.preserve_format)

                z = state['z']

                # Apply weight_decay_at_y
                if self.sf_weight_decay_at_y != 0.0:
                    z.sub_(p, alpha=lr*self.sf_weight_decay_at_y)    
                    p.sub_(p, alpha=lr*self.sf_weight_decay_at_y*(1-self.sf_momentum))

                # Unextrapolate p converting from y -> x
                p.lerp_(end=z, weight=1-1/self.sf_momentum)

                # Swap x into z buffer temporarily
                self.swap(z, p)

                # Now state['z'] is x and p is z.

        #######
        # Apply step to z
        self.base_optimizer.step()

        ######
        for group in self.param_groups:
            weight_lr_power = self.sf_weight_lr_power
            r = self.sf_r
            # tiny bit of starting LR to avoid divide by zero
            lr = max(group['lr'] * 1.0, 1e-16)
            lr_max = group['lr_max'] = max(lr, group.get('lr_max', 0))
            
            weight = (group['k']**r) * (lr_max**weight_lr_power)
            weight_sum = group['sf_weight_sum'] = group.get('sf_weight_sum', 0.0) + weight

            ckp1 = weight/weight_sum

            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                z = state['z']

                # Swap x back out of z buffer, leaving p as x
                self.swap(z, p)

                # Update x
                p.lerp_(end=z, weight=ckp1)

                # Now set p to y
                p.lerp_(end=state['z'], weight=1-self.sf_momentum)

        return loss
    
    def load_state_dict(self, state_dict: Dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
