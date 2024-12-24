# From https://github.com/pytorch/ao/blob/eab345c2268a7506355d506ebfc27b5d28e5e7d0/torchao/prototype/low_bit_optim/adam.py

from .low_bit_optim.adam import AdamW8bit, AdamW4bit, AdamWFp8

# Just a wrapper to rename
class AdamW8bitAO(AdamW8bit):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        *,
        block_size=256,
        bf16_stochastic_round=False,
    ) -> None:
        super().__init__(
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            block_size=block_size,
            bf16_stochastic_round=bf16_stochastic_round,
            is_adamw=True,
        )

    def __str__(self) -> str:
        return 'AdamW8bitAO'
    
class AdamW4bitAO(AdamW4bit):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        *,
        block_size=128,
        bf16_stochastic_round=False,
    ) -> None:
        super().__init__(
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            block_size=block_size,
            bf16_stochastic_round=bf16_stochastic_round,
            is_adamw=True,
        )

    def __str__(self) -> str:
        return 'AdamW4bitAO'
    
class AdamWfp8AO(AdamWFp8):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        *,
        block_size=256,
        bf16_stochastic_round=False,
    ) -> None:
        super().__init__(
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            block_size=block_size,
            bf16_stochastic_round=bf16_stochastic_round,
            is_adamw=True,
        )

    def __str__(self) -> str:
        return 'AdamWfp8AO'