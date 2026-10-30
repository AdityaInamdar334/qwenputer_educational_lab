"""An educational Gated DeltaNet-like state used as exact bit-addressable RAM.

This is deliberately the simple algebraic recurrence, NOT the actual Qwen3.5
kernel, quantization, RMSNorm or memory layout. Rows represent bit vectors.
"""
import torch


class DeltaRAM:
    def __init__(self, rows: int = 128, width: int = 32, *, dtype=torch.float32):
        self.state = torch.zeros(rows, width, dtype=dtype)

    @property
    def rows(self):
        return self.state.shape[0]

    def update(self, address: int, value: torch.Tensor, *, alpha: float = 1., beta: float = 1.):
        if not 0 <= address < self.rows:
            raise IndexError(address)
        value = value.to(dtype=self.state.dtype).reshape(-1)
        if value.numel() != self.state.shape[1]:
            raise ValueError("Wrong value vector width")
        k = torch.nn.functional.one_hot(
            torch.tensor(address), num_classes=self.rows
        ).to(dtype=self.state.dtype)
        decayed = alpha * self.state
        prediction = k @ decayed
        self.state = decayed + beta * torch.outer(k, value - prediction)

    def read(self, address: int) -> torch.Tensor:
        if not 0 <= address < self.rows:
            raise IndexError(address)
        q = torch.nn.functional.one_hot(
            torch.tensor(address), num_classes=self.rows
        ).to(dtype=self.state.dtype)
        return q @ self.state


def bits32(value: int) -> torch.Tensor:
    value &= 0xFFFFFFFF
    return torch.tensor([(value >> i) & 1 for i in range(32)], dtype=torch.float32)


def word32(bits: torch.Tensor) -> int:
    if bits.numel() != 32:
        raise ValueError("Expected 32 bits")
    values = bits.tolist()
    if any(v not in (0.0, 1.0) for v in values):
        raise ValueError("State no longer contains binary digits")
    ints = [int(v) for v in values]
    return sum(bit << i for i, bit in enumerate(ints))
