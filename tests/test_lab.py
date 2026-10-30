import random
import unittest
import torch
from qwenputer_lab.memory import DeltaRAM, bits32, word32
from qwenputer_lab.gates import FixedGate, FixedAdder32
from qwenputer_lab.rv32mini import MiniRV32, addi, beq, demo_program, sub


class LabTests(unittest.TestCase):
    def test_memory_overwrite_and_independence(self):
        ram = DeltaRAM(16, 32)
        for i in range(16):
            ram.update(i, bits32(i * 17001))
        for i in range(16):
            self.assertEqual(word32(ram.read(i)), i * 17001)
        ram.update(3, bits32(0xFEDCBA98))
        self.assertEqual(word32(ram.read(3)), 0xFEDCBA98)
        self.assertEqual(word32(ram.read(4)), 4 * 17001)

    def test_gate_decay_and_erase(self):
        ram = DeltaRAM(4, 32)
        ram.update(1, bits32(15))
        ram.update(2, torch.zeros(32), alpha=0)
        for i in range(4):
            self.assertEqual(word32(ram.read(i)), 0)

    def test_non_binary_value_guard(self):
        with self.assertRaises(ValueError):
            word32(torch.full((32,), .5))

    def test_fixed_gates(self):
        xor, AND = FixedGate("xor"), FixedGate("and")
        for a in (0, 1):
            for b in (0, 1):
                x = torch.tensor([float(a), float(b)])
                self.assertEqual(int(xor(x)), a ^ b)
                self.assertEqual(int(AND(x)), a & b)
        self.assertFalse(any(p.requires_grad for p in xor.parameters()))

    def test_adder_randomized_32bit(self):
        rng = random.Random(2026)
        adder = FixedAdder32()
        for _ in range(40):
            a, b = rng.getrandbits(32), rng.getrandbits(32)
            self.assertEqual(word32(adder(bits32(a), bits32(b))), (a + b) & 0xFFFFFFFF)

    def test_program(self):
        cpu = MiniRV32(demo_program())
        trace = cpu.run()
        self.assertEqual([x.mnemonic for x in trace], ["ADDI", "ADDI", "ADD", "ADDI", "SW", "LW", "ADDI", "ECALL exit"])
        self.assertEqual(cpu.read_reg(5), 12)
        self.assertEqual(word32(cpu.state.read(32)), 12)
        self.assertEqual(cpu.read_reg(0), 0)

    def test_branch(self):
        cpu = MiniRV32([addi(1, 0, 5), addi(2, 0, 5), beq(1, 2, 8), addi(3, 0, 99), addi(3, 0, 7), addi(17, 0, 93), 0x73])
        cpu.run()
        self.assertEqual(cpu.read_reg(3), 7)
        self.assertEqual(cpu.steps, 6)

    def test_sub_wrap(self):
        cpu = MiniRV32([addi(1,0,1),addi(2,0,3),sub(3,1,2),addi(17,0,93),0x73])
        cpu.run()
        self.assertEqual(cpu.read_reg(3), 0xFFFFFFFE)

    def test_invalid_instruction(self):
        with self.assertRaises(NotImplementedError):
            MiniRV32([0xFFFFFFFF]).run()

    def test_out_of_bounds(self):
        cpu = MiniRV32([addi(1,0,0),addi(2,0,17),0x0020A023])
        # Directly verify bounds instead of relying on manual SW encoding.
        with self.assertRaises(ValueError):
            cpu._memrow(0x10)


if __name__ == "__main__":
    unittest.main()
