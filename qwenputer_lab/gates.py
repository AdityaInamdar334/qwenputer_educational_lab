"""Small fixed-weight binary logic circuits in PyTorch (never trained)."""
import torch
from torch import nn


class FixedGate(nn.Module):
    """Compile a 2-input AND or XOR gate into known weights, using ReLU."""
    def __init__(self, kind: str):
        super().__init__()
        if kind == "and":
            weights = torch.tensor([[1., 1.]])
            biases = torch.tensor([-1.])
            output = torch.tensor([[1.]])
        elif kind == "xor":
            # XOR(x,y) = (x+y) - 2*ReLU(x+y-1), for binary x,y
            weights = torch.tensor([[1., 1.], [1., 1.]])
            biases = torch.tensor([0., -1.])
            output = torch.tensor([[1., -2.]])
        else:
            raise ValueError(kind)
        self.register_buffer("w1", weights)
        self.register_buffer("b1", biases)
        self.register_buffer("w2", output)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(
            torch.relu(torch.nn.functional.linear(pair, self.w1, self.b1)),
            self.w2,
        ).squeeze(-1)


class FixedAdder32(nn.Module):
    """32-bit ripple-carry addition made from fixed AND/XOR weights.

    This is deliberately NOT the Qwenputer emitter, which compiles elaborate
    RISC-V arithmetic into Qwen3.5's SwiGLU/DeltaNet layers.
    """
    def __init__(self):
        super().__init__()
        self.xor = FixedGate("xor")
        self.and_gate = FixedGate("and")

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.shape != (32,) or b.shape != (32,):
            raise ValueError("Expected two LSB-first 32-bit vectors")
        carry = torch.zeros((), dtype=a.dtype, device=a.device)
        result = []
        for i in range(32):
            axb = self.xor(torch.stack([a[i], b[i]]))
            bit = self.xor(torch.stack([axb, carry]))
            c1 = self.and_gate(torch.stack([a[i], b[i]]))
            c2 = self.and_gate(torch.stack([axb, carry]))
            # c1 and c2 cannot both be 1 for binary inputs.
            carry = c1 + c2
            result.append(bit)
        return torch.stack(result)
