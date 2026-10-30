"""Small RV32I subset teaching aid backed by DeltaRAM for storage.

IMPORTANT: Python decodes instructions and controls execution here. Unlike
Qwenputer, the CPU is NOT compiled into transformer weights. Only the bitwise
adder is fixed-weight PyTorch arithmetic; memory uses the delta recurrence.
"""
from dataclasses import dataclass
from .memory import DeltaRAM, bits32, word32
from .gates import FixedAdder32

MASK = 0xFFFFFFFF

def sext(n: int, width: int) -> int:
    sign = 1 << (width - 1)
    return (n ^ sign) - sign


def addi(rd: int, rs1: int, imm: int) -> int:
    if not -(1 << 11) <= imm < (1 << 11):
        raise ValueError("ADDI immediate must fit signed 12-bit")
    return ((imm & 4095) << 20) | (rs1 << 15) | (rd << 7) | 0x13


def add(rd: int, rs1: int, rs2: int) -> int:
    return (rs2 << 20) | (rs1 << 15) | (rd << 7) | 0x33


def sub(rd: int, rs1: int, rs2: int) -> int:
    return (0x20 << 25) | add(rd, rs1, rs2)


def sw(rs2: int, rs1: int, imm: int) -> int:
    if not -(1 << 11) <= imm < (1 << 11):
        raise ValueError("SW immediate must fit signed 12-bit")
    u = imm & 0xFFF
    return ((u >> 5) << 25) | (rs2 << 20) | (rs1 << 15) | (2 << 12) | ((u & 31) << 7) | 0x23


def lw(rd: int, rs1: int, imm: int) -> int:
    if not -(1 << 11) <= imm < (1 << 11):
        raise ValueError("LW immediate must fit signed 12-bit")
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (2 << 12) | (rd << 7) | 0x03


def beq(rs1: int, rs2: int, offset: int) -> int:
    if offset % 2 or not -(1 << 12) <= offset < (1 << 12):
        raise ValueError("BEQ offset must fit signed 13-bit and be even")
    u = offset & 0x1FFF
    return (((u >> 12) & 1) << 31) | (((u >> 5) & 63) << 25) | (rs2 << 20) | (rs1 << 15) | (((u >> 1) & 15) << 8) | (((u >> 11) & 1) << 7) | 0x63


@dataclass
class Step:
    pc: int
    mnemonic: str
    next_pc: int


class MiniRV32:
    RAM_BASE = 0x100
    RAM_WORDS = 64
    PC_ROW = 32 + RAM_WORDS

    def __init__(self, program: list[int]):
        self.rom = tuple(word & MASK for word in program)
        self.state = DeltaRAM(rows=self.PC_ROW + 1, width=32)
        self.adder = FixedAdder32()
        self.halted = False
        self.steps = 0

    def read_reg(self, reg: int) -> int:
        return 0 if reg == 0 else word32(self.state.read(reg))

    def write_reg(self, reg: int, word: int):
        if reg:
            self.state.update(reg, bits32(word))

    def _memrow(self, byte_addr: int) -> int:
        if byte_addr % 4:
            raise ValueError(f"misaligned word access: {byte_addr:#x}")
        index = (byte_addr - self.RAM_BASE) // 4
        if not 0 <= index < self.RAM_WORDS:
            raise ValueError(f"address out of RAM bounds: {byte_addr:#x}")
        return 32 + index

    def _add(self, a: int, b: int) -> int:
        return word32(self.adder(bits32(a), bits32(b)))

    def step(self) -> Step:
        if self.halted:
            raise RuntimeError("CPU is halted")
        pc = word32(self.state.read(self.PC_ROW))
        if pc % 4 or not 0 <= pc // 4 < len(self.rom):
            raise ValueError(f"invalid instruction address {pc:#x}")
        instr = self.rom[pc // 4]
        opcode = instr & 127
        rd = (instr >> 7) & 31
        f3 = (instr >> 12) & 7
        rs1 = (instr >> 15) & 31
        rs2 = (instr >> 20) & 31
        f7 = (instr >> 25) & 127
        nxt = (pc + 4) & MASK
        if opcode == 0x13 and f3 == 0:
            imm = sext(instr >> 20, 12)
            self.write_reg(rd, self._add(self.read_reg(rs1), imm))
            name = "ADDI"
        elif opcode == 0x33 and f3 == 0 and f7 in (0x00, 0x20):
            rhs = self.read_reg(rs2)
            if f7 == 0x20:
                rhs = -rhs
            self.write_reg(rd, self._add(self.read_reg(rs1), rhs))
            name = "SUB" if f7 else "ADD"
        elif opcode == 0x23 and f3 == 2:
            imm = sext(((instr >> 25) << 5) | rd, 12)
            addr = self._add(self.read_reg(rs1), imm)
            self.state.update(self._memrow(addr), bits32(self.read_reg(rs2)))
            name = "SW"
        elif opcode == 0x03 and f3 == 2:
            imm = sext(instr >> 20, 12)
            addr = self._add(self.read_reg(rs1), imm)
            self.write_reg(rd, word32(self.state.read(self._memrow(addr))))
            name = "LW"
        elif opcode == 0x63 and f3 == 0:
            imm = (((instr >> 31) & 1) << 12) | (((instr >> 7) & 1) << 11) | (((instr >> 25) & 63) << 5) | (((instr >> 8) & 15) << 1)
            imm = sext(imm, 13)
            if self.read_reg(rs1) == self.read_reg(rs2):
                nxt = self._add(pc, imm)
            name = "BEQ"
        elif instr == 0x00000073 and self.read_reg(17) == 93:
            self.halted = True
            name = "ECALL exit"
        else:
            raise NotImplementedError(f"unsupported RV32 instruction {instr:#010x}")
        self.state.update(self.PC_ROW, bits32(nxt))
        self.steps += 1
        return Step(pc, name, nxt)

    def run(self, max_steps: int = 200) -> list[Step]:
        trace = []
        for _ in range(max_steps):
            trace.append(self.step())
            if self.halted:
                return trace
        raise TimeoutError("program did not halt within step budget")


def demo_program():
    return [
        addi(1, 0, 7),
        addi(2, 0, 5),
        add(3, 1, 2),
        addi(4, 0, MiniRV32.RAM_BASE),
        sw(3, 4, 0),
        lw(5, 4, 0),
        addi(17, 0, 93),
        0x00000073,
    ]


if __name__ == "__main__":
    cpu = MiniRV32(demo_program())
    for step in cpu.run():
        print(f"pc={step.pc:04x} {step.mnemonic:10s} next={step.next_pc:04x}")
    print("x5:", cpu.read_reg(5))
    print("RAM[0x100]:", word32(cpu.state.read(32)))
