"""Small CPU benchmarks of educational mechanics; no Qwen model is loaded."""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

from qwenputer_lab.memory import DeltaRAM, bits32, word32
from qwenputer_lab.gates import FixedAdder32
from qwenputer_lab.rv32mini import MiniRV32, addi, add


def trial(seed: int, operations: int):
    rng = random.Random(seed)
    ram = DeltaRAM(rows=64, width=32)
    reference = [0] * 64
    start = time.perf_counter()
    for _ in range(operations):
        address = rng.randrange(64)
        value = rng.getrandbits(32)
        ram.update(address, bits32(value))
        reference[address] = value
        check_address = rng.randrange(64)
        assert word32(ram.read(check_address)) == reference[check_address]
    memory_s = time.perf_counter() - start
    assert all(word32(ram.read(i)) == reference[i] for i in range(64))

    adder = FixedAdder32()
    start = time.perf_counter()
    for _ in range(operations):
        a, b = rng.getrandbits(32), rng.getrandbits(32)
        assert word32(adder(bits32(a), bits32(b))) == (a + b) & 0xFFFFFFFF
    adder_s = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(16):
        a, b = rng.randrange(0, 1000), rng.randrange(0, 1000)
        program = [addi(1,0,a), addi(2,0,b), add(3,1,2), addi(17,0,93), 0x73]
        cpu = MiniRV32(program)
        cpu.run()
        assert cpu.read_reg(3) == a + b
    cpu_s = time.perf_counter() - start
    return {"seed":seed,"memory_operations":operations,"memory_s":memory_s,
            "fixed_adder_cases":operations,"adder_s":adder_s,"cpu_programs":16,"cpu_s":cpu_s,
            "memory_errors":0,"adder_errors":0,"cpu_errors":0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--operations', type=int, default=250)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--output', default='results/benchmark.json')
    args = ap.parse_args()
    records = [trial(2026 + i, args.operations) for i in range(args.seeds)]
    summary = {k: {"median": statistics.median(r[k] for r in records), "min":min(r[k] for r in records), "max":max(r[k] for r in records)} for k in ('memory_s','adder_s','cpu_s')}
    output = {"description": "Educational PyTorch CPU-only mechanics benchmark. Not Qwenputer performance.",
              "operations_per_seed":args.operations,"seeds":args.seeds,
              "records":records,"timing_s":summary}
    path = Path(args.output)
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
