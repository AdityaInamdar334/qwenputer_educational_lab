# Qwenputer: Educational DeltaNet RAM and Compiled Logic Lab

A small, independent CPU-only PyTorch project explaining two ideas behind [Qwenputer](https://github.com/divagr18/qwenputer): an addressable recurrent memory using the delta rule, and binary logic built from fixed weights rather than learned parameters.

This repository is **not the Qwenputer implementation**. It does not include pretrained Qwen weights, and its minimal RV32I subset executes instruction decoding and control flow in Python. The original project compiles instruction execution into 11 Qwen3.5-compatible neural layers. Read its [design notes](https://github.com/divagr18/qwenputer/blob/main/experimental/riscv/docs/rv32.md) to understand that distinction.

## What you can run

- `qwenputer_lab/memory.py`: one-hot state-matrix writes and reads, `alpha` decay, `beta` write strength, 32-bit bit-vector storage.
- `qwenputer_lab/gates.py`: exact fixed-weight AND and XOR gates using ReLU and a 32-bit ripple-carry adder.
- `qwenputer_lab/rv32mini.py`: a small RV32I subset with `ADDI`, `ADD`, `SUB`, `LW`, `SW`, `BEQ`, and exit-style `ECALL`.
- `tests/test_lab.py`: ten correctness tests, including wraparound arithmetic, branch behavior, and memory isolation.
- `benchmark.py`: CPU-only reproducibility experiment with three seeds.
- `results/benchmark.json`: benchmark output from one local run. Timing values are machine-specific.
- `ARTICLE.md`: full companion technical article.

## Setup

Python 3.10+ and PyTorch are needed. No model weights or GPU are required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m qwenputer_lab.rv32mini
python benchmark.py --seeds 3 --operations 200 --output results/benchmark.json
```

On Windows PowerShell, activate using `.venv\Scripts\Activate.ps1`.

## Sample result

The example program adds 7 and 5, stores the result in simulated RAM, loads it back, then halts.

```text
x5: 12
RAM[0x100]: 12
```

For the local test run, all ten unit tests passed. Across three seeds, the benchmark checked 200 random memory accesses, 200 random 32-bit additions, and 16 short CPU programs per seed with zero detected mismatches. These figures do not establish correctness outside the tested cases, and none are Qwenputer GPU throughput benchmarks.

## Mathematical mechanism

Each memory head carries a matrix called `state`. For a one-hot key `k`, `alpha=1`, and `beta=1`, the delta update becomes:

```python
old = state
prediction = k @ old
state = old + torch.outer(k, value - prediction)
```

The selected row is overwritten exactly for our binary bit-vector representation, while all other rows stay unchanged. A one-hot query reads that row. Changing `alpha` below one causes decay; changing `beta` below one blends old and new information, which no longer behaves like ordinary overwriting RAM.

## What this lab does not prove

- It does not implement Qwen3.5's SwiGLU activations, RMSNorm scaling, DeltaNet CUDA kernels, or bf16 numerical proof obligations.
- It does not compile the RV32I decoder into model weights. The Python interpreter remains responsible for decoding and scheduling.
- It implements only a small RV32I subset, not RV32IM, Zba, MicroPython, or the original project's big-integer coprocessor.
- It is not an instruction-set compliance suite or a security sandbox.
- It does not benchmark a pretrained language model, token output, or GPU execution.

## Original sources

- [Divyansh Agrawal's original Qwenputer write-up](https://divagr.com/blog/qwenputer)
- [Original source repository](https://github.com/divagr18/qwenputer)
- [Detailed RISC-V design](https://github.com/divagr18/qwenputer/blob/main/experimental/riscv/docs/rv32.md)
- [Gated Delta Networks](https://arxiv.org/abs/2412.06464)
- [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174)

Educational implementation and article assembled in October 2026. No claim of affiliation with Qwenputer's author or Qwen.
