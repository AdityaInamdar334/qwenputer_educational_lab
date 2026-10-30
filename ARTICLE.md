# A RISC-V CPU Inside Qwen's Weights: How Qwenputer Turns Gated DeltaNet Into Exact RAM

## A technical deep dive into compiling deterministic circuits into a language model, constructing addressable recurrent memory, executing instructions through token generation, and the security implications of computation hidden inside neural state

A language model asks a calculator to multiply two large integers.

Ordinarily, that request crosses a familiar boundary. The model generates a tool call. An external runtime executes the calculation. The result is returned to the model, which incorporates it into its answer.

Qwenputer makes that boundary surprisingly difficult to locate.

In an experimental Qwen3.5-4B checkpoint, the calculator is backed by a real 32-bit RISC-V processor. But there is no external Python interpreter executing the calculation. Instead, eleven additional neural-network layers perform the processor's operations. The processor stores its registers, program counter, code, and data in the recurrent state of Gated DeltaNet heads. Its arithmetic logic was not learned by gradient descent. A compiler constructed the additional weights directly.

The system can run C programs and a standalone MicroPython environment. It can execute millions of instructions, and the author reports matching the instruction traces against reference machines without discrepancies in tested runs. Meanwhile, the original language-model weights are preserved. [1][2][3]

If that sounds like an LLM pretending to be a CPU, it is not. The model is not merely predicting what an instruction *would* do. There is a specified circuit embedded in the computation graph whose state transitions implement the operations.

There is also a crucial caveat. Qwenputer is not a bare checkpoint that runs a full operating system on its own. Its host-side driver loads firmware into recurrent state, controls the execution path, and prevents internal trace tokens from confusing the pretrained language model. The instruction execution itself is in the compiled circuit, but the experiment still requires software orchestration. [2][3]

That separation makes Qwenputer interesting to far more than people who like unusual model demos. It exposes a connection between linear recurrent attention, fast-weight memory, compiler construction, deterministic arithmetic, and the limits of monitoring language models by reading their visible text.

In this article, we will examine the author's implementation and source code, derive the memory mechanism in plain language, and build a small educational PyTorch laboratory with the same conceptual ingredients. We will also run correctness experiments rather than assume that numerical circuits are exact simply because they use deterministic weights.

The laboratory is deliberately **not** a reproduction of Qwenputer's full weight compiler. It implements addressable DeltaNet-style state, fixed-weight logic gates, and a small RISC-V instruction subset with Python handling instruction decoding. We will be explicit about every boundary.

## Table of Contents

1. Why putting a CPU inside a language model is a different idea from tool calling
2. What Qwenputer actually changes in Qwen3.5-4B
3. The critical ingredient: Gated DeltaNet as a recurrent state machine
4. Why a one-hot key can behave like a RAM address
5. Exactness requires choosing the representation carefully
6. How the project compiles circuits into neural-network weights
7. The eleven-stage RISC-V execution pipeline
8. ROM, RAM, registers, and the program counter
9. Why each instruction can require multiple generated tokens
10. How Qwen communicates with an internal calculator
11. What the original benchmarks establish
12. Why deterministic CPU results do not guarantee perfect model answers
13. Reproduction Part I: Build addressable memory in PyTorch
14. Reproduction Part II: Compile logic gates without training
15. Reproduction Part III: Execute a small RV32I program
16. Validate the memory and arithmetic invariants
17. Our controlled CPU-only benchmark
18. How to run the full Qwenputer checkpoint
19. What the design costs in compute and memory
20. Persistent memory and the experimental in-weight filesystem
21. Security: invisible computation is still computation
22. Unanswered research questions and useful extensions
23. Conclusion
24. References and source code

---

## 1. Why Putting a CPU Inside a Language Model Is Different From Tool Calling

The first distinction is between *using* a computer and *containing* the computer's logic.

Consider a conventional AI assistant that can access a calculator. The assistant emits something like:

```text
<tool_call>
calc("123456789 * 987654321")
</tool_call>
```

The inference server recognizes this as a request, stops normal decoding, calls an external calculator, and inserts its result into the conversation.

The arithmetic is reliable because the external computer implements arithmetic, not because the LLM has discovered a better multiplication algorithm.

This is often the correct engineering solution. We already have optimized CPUs, interpreters, and carefully sandboxed tools. There is no inherent practical advantage to recreating them inside model layers.

But now consider a different question: can a pretrained model's own architecture support a deterministic, general-purpose computational substrate?

In Qwenputer, Divyansh Agrawal answers this by constructing a processor as extra Qwen3.5-compatible layers. Instead of letting the host service evaluate the calculation, the model's forward computations evolve the processor's state. [1][2]

That is not merely a change in API packaging.

An ordinary tool is external to the neural network. The CPU in Qwenputer occupies neural layers and recurrent state. The compiler-generated weights establish specific logic and memory behavior. The token stream provides a clock-like sequence of operations.

The result has features of a hybrid system: a trained language model coupled to a hand-compiled deterministic machine inside one checkpoint.

The central architectural question is therefore not whether neural networks are better CPUs. They are not generally competitive with conventional silicon at this task. The question is whether a recurrent neural layer can be programmed with exact state transitions and composed into a verified computation device.

## 2. What Qwenputer Actually Changes in Qwen3.5-4B

Qwen3.5 uses a hybrid architecture. In the 4B model, its 32 language-model layers follow a repeating pattern of three Gated DeltaNet layers followed by one full-attention layer. Gated DeltaNet provides a fixed-size recurrent state for part of the context processing, while periodic full-attention layers preserve another mechanism for token interaction. [9]

Qwenputer prepends eleven circuit layers to that original backbone.

The original Qwen weights are not retrained to implement RISC-V. The new circuit layers have compiler-generated weights, while the pretrained model remains responsible for interpreting the user's request and producing ordinary language where the circuit does not own generation. [2][3]

The CPU supports RV32IM, the 32-bit base integer instruction set and integer multiply/divide extension, together with Zba address-generation instructions and a custom big-integer unit. The author's implementation is documented in the `experimental/riscv/` directory. [3]

The project uses a modified vocabulary and additional hidden dimensions to represent circuit events. The detailed P3 design document reports 478,656 vocabulary entries and a hidden width of 3,968 in the merged configuration, with roughly 18.7 GB of bfloat16 checkpoint storage. The author describes the checkpoint as approximately 18.8 GB in the project overview. [3]

Those numbers matter for two reasons.

First, the added circuit is sparse in its logical structure but stored inside dense neural-network parameter tensors, so the physical checkpoint can be large even when few compiled weights are nonzero.

Second, attaching a deterministic circuit to a trained model is not free. The method increases model size and requires careful routing between the two components.

A helpful mental model is to imagine that the forward pass begins with a small computer and then continues through the familiar language model when appropriate.

```text
User request
    |
    v
Qwen token generation
    |
    v
Compiled circuit layers
    |   register file
    |   code and data memory
    |   ALU and branch logic
    |   output control
    v
Original Qwen3.5 layers
    |
    v
Ordinary language token or circuit-controlled result
```

This is conceptual, not a literal picture of every kernel call. The implementation has special execution handling for circuit tokens, and the original Qwen layers do not process those internal trace positions. [3]

## 3. The Critical Ingredient: Gated DeltaNet as a Recurrent State Machine

A standard full-attention decoder usually retains past keys and values in a KV cache. As the sequence grows, that cache generally grows with it.

Gated DeltaNet instead maintains a recurrent state matrix for each head. The state is updated as new tokens arrive. Its dimensions do not need to grow with the number of processed tokens.

The Gated DeltaNet research combines adaptive forgetting with the delta update rule. The forgetting gate determines how much existing state survives, while the delta update modifies the mapping from keys to values. [7]

For our educational derivation, think of a memory matrix with a number of address rows and a number of value columns.

At a new step, the model first scales the old matrix by a gate, which we will call `alpha`.

It then looks up what value is currently associated with the incoming key.

Next, it calculates the difference between the desired value and the current value.

Finally, it writes a fraction of that difference back to the memory, controlled by a write strength called `beta`.

In pseudocode:

```python
old = alpha * state
prediction = key @ old
correction = value - prediction
state = old + beta * outer(key, correction)
output = query @ state
```

This captures the essential delta-rule mechanism used in the demonstration. It is not intended as a line-for-line description of every normalization, projection, convolution, or numerical operation in the production Qwen3.5 kernel.

Notice what the correction accomplishes. If a key already maps to the correct value, the correction is zero. If the stored association is wrong, the write changes it toward the intended value.

Schlag, Irie, and Schmidhuber had already identified an important relationship between linear transformers and *fast weight programmers* in 2021. Their work showed how recurrent attention can manipulate finite associative memory through write-like operations. [8]

Qwenputer builds on that lineage but pushes the engineering much further: it arranges state updates so precisely that they become random-access memory operations, suitable for a CPU.

## 4. Why a One-Hot Key Can Behave Like a RAM Address

The critical setting is simple enough to explain without algebraic notation.

Set the forgetting gate to 1, so previous rows are not decayed.

Set write strength to 1, so the selected association is completely replaced rather than partially blended.

Choose a one-hot key, which has exactly one active element.

If the key selects row 7, the old prediction is simply the value stored in row 7. The correction is the requested new value minus that old value. Adding the correction overwrites row 7 and leaves the other rows alone.

A one-hot query for row 7 returns its contents.

```text
Before write
row 0: [0, 1, 0, 0]
row 1: [1, 1, 0, 0]
row 2: [0, 0, 0, 0]
row 3: [1, 0, 0, 1]

Key selects row 2
Value to write: [1, 0, 1, 1]

After write
row 0: [0, 1, 0, 0]
row 1: [1, 1, 0, 0]
row 2: [1, 0, 1, 1]
row 3: [1, 0, 0, 1]
```

Read row 2, and you recover the written vector.

The powerful observation is that the state's physical size does not grow just because another token has been processed.

The memory is finite, however. One-hot addressing works only for the number of rows and columns physically available in the recurrent state, and capacity must be deliberately partitioned across heads.

The author maps address bits to key heads and rows, with separate heads for different forms of machine state. The Qwenputer design uses multiple heads for RAM and program storage, not an unlimited single head. [2][3]

## 5. Exactness Requires Choosing the Representation Carefully

It is easy to say that a deterministic matrix operation is exact. It is much harder to make that statement survive real accelerator kernels and reduced-precision formats.

The most important design decision is to represent CPU words as bits rather than as arbitrary large floating-point integers.

A float32 value cannot represent every 32-bit integer exactly. Its significand has finite precision. If we stored an entire 32-bit RISC-V word as one scalar, apparently ordinary arithmetic could lose lower bits.

Instead, each bit is stored separately as an exact binary value, 0 or 1. Those values are exactly representable. Boolean operations can also be implemented using carefully chosen arithmetic on these small values.

Our tutorial follows that representation because it makes correctness easy to test.

The original project goes considerably deeper. Its compiler targets Qwen3.5's SwiGLU and recurrent mechanisms and uses scaling, controlled saturation, and specialized layouts so that desired logic functions survive the model's nonlinearities and normalization. [4][5][6]

The compiler's intermediate representation explains that it constructs feed-forward terms using a ReLU-like abstraction, then emits them through a scaled SiLU configuration in which the reachable circuit values make the behavior effectively exact.

It also handles recurrent normalization using reference columns and ballast values, and checks numerical constraints during compilation. [4][5]

These are not cosmetic implementation details.

Qwen3.5 is not a network of idealized Boolean gates. It has normalization, finite-precision arithmetic, nonlinear activations, and GPU kernels whose numerical behavior can depend on how operations are grouped.

The Qwenputer design documentation even records a case where multi-hot reads introduced small floating-point residues. The author changed the relevant reads to one-hot or specially constrained two-hot patterns. [3]

That is why the project is a serious compiler exercise, not just a theoretical observation that matrices can store numbers.

## 6. How the Project Compiles Circuits Into Neural-Network Weights

Compiling a CPU into a transformer is an unusual compiler problem because the target instruction set is not a CPU instruction set at all.

The target consists of operations that Qwen3.5's layers can perform: linear projections, SwiGLU feed-forward units, recurrent key-value writes and reads, and residual-stream transformations.

The source, on the other hand, is a circuit description representing RISC-V instruction execution.

The original code organizes the compiler around three notable components:

- `chipformer/q35/ir.py` describes the circuit graph, values, heads, and scheduling constraints.
- `chipformer/q35/prims.py` constructs exact primitives such as Boolean conditions, comparisons, and bit-based operations.
- `chipformer/q35/emit.py` allocates residual slots and turns the scheduled graph into Qwen-compatible weights. [4][5][6]

The source-to-target translation is conceptually similar to hardware synthesis.

An instruction decoder needs to identify the operation and register fields. The register file needs reads and writes. The ALU needs bitwise functions, addition, shifts, multiplication, and division. Control flow needs a program counter and conditional branches.

The compiler decomposes those operations into stages that fit inside the available neural layers.

A simplified pipeline looks like this:

```text
RISC-V instruction semantics
          |
          v
Small logic and arithmetic primitives
          |
          v
Dependency graph and layer scheduling
          |
          v
Residual-stream and memory-head allocation
          |
          v
Constructed linear and nonlinear weights
          |
          v
Qwen3.5-compatible circuit layers
```

No gradient-based search is required to discover an addition algorithm. The compiler already knows the algorithm and produces weights that realize it.

This changes what it means to call the result *neural computation*. The substrate is a neural network, but the computation is largely a hand-designed digital circuit translated into that substrate.

## 7. The Eleven-Stage RISC-V Execution Pipeline

The final processor is implemented in eleven neural layers, and its stage boundaries are constrained by data dependencies. [3]

The first layer reads machine-control state such as the program counter and instruction-record bookkeeping. It also prepares representations used by later stages.

The second prepares instruction addresses and related values.

The third and fourth fetch predecoded instruction information from code memory and perform additional decoding.

The fifth reads registers and begins arithmetic operations.

The sixth continues addition, shifting, logical operations, and multiplication-related computations.

The seventh and eighth include data-memory access and further arithmetic transformations.

The ninth accesses read-only data memory and performs later multiplication and division work.

The tenth resolves remaining division and destination-register selection logic.

The eleventh selects the circuit's output token and finalizes the step.

The exact operations depend on the instruction. Not every stage performs meaningful work for every opcode.

The essential engineering point is dependency depth.

A compiler can parallelize independent operations, but the result of an instruction decode must exist before dependent register selection, and some arithmetic stages cannot be collapsed without changing the circuit design.

The project documentation reports that the eleven-layer depth is governed by those dependencies, even though the representation contains many operations that can execute in parallel within a layer. [3]

## 8. ROM, RAM, Registers, and the Program Counter

A CPU requires more than an adder.

It needs instruction storage, writable data, a register file, and a program counter. It also needs a defined way to halt and report faults.

In Qwenputer, the recurrent heads provide the storage substrate for these components.

The project documents a 32 KB RAM region, a 128 KB data-ROM region, and a 256 KB instruction-ROM region in its P2 configuration, along with register, quotient, and bookkeeping heads. Code and read-only data are loaded as part of the machine's initialized state. [3]

The instruction ROM is not simply a raw stream of opcodes that the language model must decipher from natural language.

The compiler can preprocess instruction fields and store useful decoded information, reducing the neural work needed during every fetch. It also folds some static facts, such as known branch targets, into the prepared code image.

This is a familiar compiler technique: spend work once at load time so the runtime path can be simpler.

The ordinary integer registers have RISC-V semantics, including a register that must always read as zero. Loads and stores access the simulated machine's address space, not arbitrary host process memory.

That distinction will become important when we discuss security.

## 9. Why Each Instruction Can Require Multiple Generated Tokens

A normal CPU uses a clock cycle or a sequence of micro-operations to advance its state.

Qwenputer uses token positions as the progression mechanism.

The project gives internal record types names such as `ADDR`, `VAL`, and `STEP`. The address and value portions carry the information needed for writes, while record-completion tokens mark instruction progress. The actual trace encoding is more detailed, with specialized record types and payload widths. [1][2][3]

A simple instruction may need only a small number of circuit positions. A more complex arithmetic operation or memory transaction may need several.

The P2 design documentation reports workloads averaging around 1.2 to 1.5 trace tokens per retired instruction in selected C and MicroPython programs, with variance by instruction mix. More elaborate instructions may emit intermediate records. [3]

This design has an unusual consequence.

The execution trace is a token sequence, but it is not ordinary chain-of-thought text. It is a machine-readable internal execution protocol for a compiled circuit.

When the circuit is active, predicting the next token is equivalent to advancing a specifically designed state machine.

## 10. How Qwen Communicates With an Internal Calculator

The language-model interface uses a clever shortcut.

Qwen3.5 already understands tool-call syntax, so the author reused that familiar interaction pattern.

The language model generates a `calc` call. The compiled circuit captures its contents. A small firmware program interprets the requested arithmetic, and the circuit emits a tool-response-shaped sequence that the language model can read. [2][3]

```text
User: Compute a large product.
        |
        v
Qwen writes a calc-shaped call
        |
        v
Compiled circuit captures input
        |
        v
RISC-V firmware computes the result
        |
        v
Circuit emits a response-shaped token sequence
        |
        v
Qwen continues with the answer
```

Nothing in the host application needs to call a separate calculator to obtain the answer.

But the distinction between computation and orchestration remains essential. `RvChat` loads the firmware into the circuit state after the prompt and routes internal trace tokens so that the pretrained language model does not process them as ordinary text. [3]

The design document describes an input capture window with limits. It also documents the `calc` firmware's operations, including integer arithmetic, exact rationals in a later revision, and custom big-number operations.

This is an engineered interface with constraints, not a general-purpose shell magically hidden behind the model's vocabulary.

## 11. What the Original Benchmarks Establish

The author reports comparisons against base Qwen3.5-4B and against the same base model using an external Python tool. These are project-reported results, not benchmarks reproduced in this tutorial. [2][3]

On a 40-question DeepMind big-multiplication set, the original model scored 55%, while Qwenputer scored 97.5%. The external Python-tool condition scored 100%.

On the big-division set, the base model scored 72.5%, Qwenputer 97.5%, and the Python-tool condition 95%.

On longer combined multiplication and division problems, the repository's current headline results give Qwenputer 100%, base Qwen 90%, and the Python-tool condition 97.5%. The detailed engineering log explains that the result depends on the revised calculator behavior and its exact-rational semantics. [3]

The project also reports 97.5% on large addition and subtraction, compared with 87.5% for the base model.

These numbers demonstrate something narrower and more interesting than a generic claim that an embedded CPU beats Python.

The model can already make use of a tool-shaped interface. Whether the underlying computation comes from the host or the embedded processor, the quality of the final answer still depends on the model using the tool correctly.

The CPU's deterministic internal computation can be correct even when the language model chooses the wrong call, enters a loop, or misstates the final value.

The author also reports replaying approximately 230,000 circuit-owned output positions against a reference evaluator with zero mismatches, and large collections of free-running instruction tests matching the reference machine or QEMU under the tested programs. [2][3]

These checks support the correctness of the compiled circuit within the evaluated scope. They are not a formal proof of correctness for every program, numerical backend, or future checkpoint variation.

## 12. Why Deterministic CPU Results Do Not Guarantee Perfect Model Answers

There are two distinct error sources in this hybrid system.

**Circuit-level error:** The compiled machine produces the wrong result for an instruction or emits the wrong output token.

**Agent-level error:** Qwen chooses the wrong operation, formats the request incorrectly, fails to use the correct result, or continues after receiving an accurate result and produces an incorrect final answer.

The author's detailed logs are unusually candid about that distinction. In the reported arithmetic evaluations, some failed answers occurred even after the machine had supplied the correct intermediate result. Other failures involved repeated calculator calls or limitations of earlier firmware semantics. [3]

This is exactly the same evaluation problem we encounter with conventional agents.

Giving an agent a correct tool does not make the entire agent pipeline correct.

For a rigorous benchmark, we should separately measure instruction conformance, trace fidelity, tool-response correctness, tool-call selection, final-answer correctness, execution latency, and memory footprint.

If these measures are collapsed into one success percentage, the result is hard to interpret.

## 13. Reproduction Part I: Build Addressable Memory in PyTorch

We will now build the smallest useful piece of the architecture.

This reproduction runs on a CPU and does not require downloading Qwen3.5. It provides a learning environment for the memory and fixed-weight arithmetic mechanisms.

Install PyTorch, then define a memory matrix:

```python
import torch
import torch.nn.functional as F

class DeltaRAM:
    def __init__(self, rows=128, width=32):
        self.state = torch.zeros(rows, width)

    def write(self, address, value):
        key = F.one_hot(
            torch.tensor(address),
            num_classes=self.state.shape[0],
        ).float()

        old_prediction = key @ self.state
        self.state = self.state + torch.outer(
            key,
            value - old_prediction,
        )

    def read(self, address):
        query = F.one_hot(
            torch.tensor(address),
            num_classes=self.state.shape[0],
        ).float()
        return query @ self.state
```

This is the special case of the delta update with forgetting gate and write strength both fixed at 1.

A short example demonstrates overwrite semantics:

```python
ram = DeltaRAM(rows=8, width=4)

ram.write(2, torch.tensor([1., 0., 1., 1.]))
print(ram.read(2))

ram.write(2, torch.tensor([0., 1., 1., 0.]))
print(ram.read(2))

print(ram.read(1))
```

The first read returns the first vector. The second returns the replacement. The unrelated row remains zero.

An ordinary PyTorch tensor assignment would be faster for this small experiment. We deliberately use the delta-rule form to expose the algebra behind the memory operation.

The complete companion implementation also supports decay and partial writes so that you can see how changing the gate or write strength changes the meaning of memory.

Try changing the gate from 1 to 0.9.

You will no longer have ordinary persistent RAM. Existing state decays on every update. That may be useful for learned contextual memory, but it is not the behavior a conventional CPU register file needs.

## 14. Reproduction Part II: Compile Logic Gates Without Training

Memory is only half of a processor.

The other half is deterministic logic.

Neural networks are commonly trained to approximate functions. But some small Boolean functions can be implemented directly with known weights.

For binary inputs, an AND gate can be constructed as the positive part of the sum of its two inputs minus one.

An XOR gate can be constructed by taking the sum of the inputs and subtracting twice the same AND gate.

Here is a tiny fixed-weight module:

```python
import torch
from torch import nn

class FixedXOR(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "w1",
            torch.tensor([[1., 1.], [1., 1.]]),
        )
        self.register_buffer("b1", torch.tensor([0., -1.]))
        self.register_buffer("w2", torch.tensor([[1., -2.]]))

    def forward(self, pair):
        hidden = torch.relu(
            torch.nn.functional.linear(pair, self.w1, self.b1)
        )
        return torch.nn.functional.linear(hidden, self.w2).squeeze(-1)
```

There is no optimizer.

No dataset is needed to discover XOR.

The weights implement a known function for the discrete inputs 0 and 1.

From XOR and AND, we can build a full adder.

First compute the XOR of the two operand bits. Then XOR that value with the incoming carry to get the output bit. The outgoing carry is the OR of the two relevant AND conditions.

Repeat this process for 32 positions to build a ripple-carry 32-bit adder.

The companion project implements exactly that in `FixedAdder32` and checks randomized 32-bit additions against Python's integer arithmetic with wraparound at 32 bits.

This approach illustrates *compilation into weights*.

It does not reproduce Qwenputer's actual emitter. Qwenputer targets SwiGLU, Gated DeltaNet heads, token embeddings, normalization, and a dense checkpoint format. Our small network uses explicit ReLU-based fixed gates and ordinary PyTorch tensor operations, making the logic much easier to inspect.

## 15. Reproduction Part III: Execute a Small RV32I Program

The next step combines state and logic.

The companion lab implements a minimal RISC-V instruction subset with `ADDI`, `ADD`, `SUB`, `LW`, `SW`, `BEQ`, and one exit-style `ECALL`.

The decoder is Python code. The integer addition runs through the fixed-weight 32-bit adder. The register file, data RAM, and program counter are stored as 32-bit vectors in DeltaRAM rows.

Unlike Qwenputer's full processor, this teaching interpreter does not compile instruction decoding or program control into neural weights. It does not implement the complete RV32I or RV32IM ISA.

A sample program performs the following operations:

```text
ADDI x1, x0, 7
ADDI x2, x0, 5
ADD  x3, x1, x2
ADDI x4, x0, 256
SW   x3, 0(x4)
LW   x5, 0(x4)
ADDI x17, x0, 93
ECALL
```

It adds 7 and 5, stores the result in simulated RAM at byte address `0x100`, reads it back into register `x5`, and halts.

Run the program with:

```bash
python -m qwenputer_lab.rv32mini
```

Its final output includes:

```text
x5: 12
RAM[0x100]: 12
```

This exercise shows exactly where the implementation boundaries are.

The memory lives in delta-rule recurrent state.

The arithmetic uses fixed neural weights.

The instruction scheduler and decoder are ordinary Python.

A faithful Qwenputer reproduction would have to replace those Python operations with a compiled graph of circuit primitives, schedule the graph into compatible Qwen layers, encode internal tokens, and test every step against the ISA reference.

That is a much larger engineering project, and the original repository already contains the relevant compiler and validation harness.

## 16. Validate the Memory and Arithmetic Invariants

Correctness matters more than a compelling demonstration screenshot.

The companion tests cover the following properties.

**One-hot overwrite:** Writing one row replaces its contents without changing unrelated rows.

**Read after write:** Reading the selected address retrieves the most recently written 32-bit value.

**Explicit clearing:** Setting the forgetting gate to zero clears the represented state.

**Fixed gate truth tables:** AND and XOR agree with their Boolean definitions for every possible pair of input bits.

**32-bit arithmetic:** Randomized 32-bit additions agree with a standard integer reference modulo 2 to the power of 32.

**Register semantics:** RISC-V register `x0` remains zero even when a program attempts to modify it.

**Memory and control flow:** The small program executes loads, stores, branches, and a halt correctly in the supported subset.

**Failure behavior:** Unsupported instructions and out-of-range accesses are rejected.

The test suite contains ten tests. It is not a formal verification of the full RISC-V ISA, but it gives readers concrete examples of the invariants a larger compiler must preserve.

For readers extending this project, one particularly useful next step is differential testing. Generate random short instruction streams, execute each stream in the educational CPU and a separate reference interpreter, and compare the full architectural state after every instruction.

The earlier the mismatch appears, the easier it is to identify whether the fault originated in decoding, address arithmetic, memory layout, or an ALU operation.

## 17. Our Controlled CPU-Only Benchmark

I ran the educational lab locally with PyTorch 2.10.0+cpu.

The benchmark uses three deterministic random seeds. For each seed, it performs 200 randomly addressed 32-bit memory writes and checks reads against a Python reference array. It also checks 200 32-bit fixed-weight additions and runs sixteen short CPU programs whose final register values are compared against Python arithmetic.

Across all three seeds, the benchmark found **zero memory, addition, or CPU-program mismatches**.

On this specific CPU environment, the median elapsed time for 200 memory operations and associated checks was approximately **13.3 milliseconds**.

The median for 200 additions using the deliberately unoptimized PyTorch ripple-carry circuit was approximately **0.455 seconds**.

The median for sixteen short RV32 subset programs was approximately **0.168 seconds**.

These timing figures are a diagnostic of our educational Python/PyTorch implementation, not of Qwenputer. They do not isolate GPU kernel execution, include a large Transformer, or measure token-generation throughput. They should not be extrapolated to compare Qwenputer with an actual RISC-V CPU.

The especially slow fixed-weight adder illustrates a useful principle: a mathematically compact circuit can be expensive when evaluated through many tiny eager PyTorch operations. Kernel fusion, batching, layout, and runtime implementation often matter more than the operation count on paper.

The benchmark is saved in `results/benchmark.json` and can be rerun using the companion project.

## 18. How to Run the Full Qwenputer Checkpoint

Readers with suitable GPU hardware can run the author's actual implementation rather than our teaching code.

The repository documents a workflow using `uv`, the Hugging Face CLI, and a checkpoint that can fit on a 24 GB GPU in the supported configuration. [2][3]

The author's published quickstart is:

```bash
git clone https://github.com/divagr18/qwenputer
cd qwenputer/experimental/riscv
uv sync --extra hf

hf download divagr1925/Qwen3.5-4B-Qwenputer \
  --local-dir qwenputer-rv

uv run python -m chipformer.rv.chat qwenputer-rv \
  --firmware tests/data/rv/calc-bridge.elf \
  --ask "Use calc: what is 123456789 * 987654321?"
```

Check the project's current README and environment requirements before attempting a full run. Model and dependency versions may change, and downloading the checkpoint is a substantial operation.

Most importantly, this command runs the author's real weight-compiled system. It is a different experiment from the small CPU-only tutorial in our companion repository.

## 19. What the Design Costs in Compute and Memory

Qwenputer is not designed to win a speed contest against an ordinary calculator.

Its value is architectural demonstration.

The project author reports that the ordinary Hugging Face circuit execution path ran roughly 250 internal circuit tokens per second on an NVIDIA L4, while an optimized Triton-based executor reached about 3,200 circuit tokens per second in the measured configuration. [3]

This executor follows the same circuit's arithmetic but uses more efficient GPU operations for its live structure. The author validates its outputs against the reference representation.

One calculator call can require thousands of internal positions. Even when the internal processor executes correctly, generating the surrounding natural-language response remains expensive.

A conventional CPU performs 32-bit addition in a tiny fraction of the time taken by this system.

A model-based implementation pays for embedding, recurrent operations, output selection, orchestration, and the underlying pretrained model's generation.

There is also a memory cost. The project enlarges hidden-state and vocabulary dimensions, adds layers, and stores sparse logical structure inside parameter tensors that may be dense in physical representation.

The source documentation shows that the compiled machine can have far fewer nonzero weights than its dense storage footprint might suggest. [3][5]

This tradeoff is common in systems research. A new computational capability can be compelling even when its first implementation is not commercially efficient.

But we should resist the temptation to turn a proof of concept into a deployment recommendation without measuring throughput, total cost, isolation behavior, and failure handling.

## 20. Persistent Memory and the Experimental In-Weight Filesystem

Once a recurrent head can behave like RAM, it becomes natural to ask whether the same substrate can store longer-lived structured data.

The author built an experimental filesystem variant that uses recurrent state to store a directory and chains of text rows. The public description reports a directory with 1,024 hash slots, support for up to 512 files, and storage on the order of 53,000 tokens in its tested configuration. [1][2]

The important nuance is persistence.

The recurrent state exists during inference, but it is not automatically durable across restarts. To make it persistent, the system must save a snapshot and restore it in a later session.

This is different from storing information permanently in the pretrained weights. The circuit weights define how memory operations work. The saved recurrent state contains the changing data.

That separation resembles the distinction between a CPU's logic and its RAM contents, even though both are represented using machinery associated with a language model.

The filesystem variant remains experimental and is distinct from the main RV32 calculator checkpoint. Its reported retrieval and memory benchmarks should not be treated as demonstrated behavior of the base Qwenputer model without the filesystem extension.

## 21. Security: Invisible Computation Is Still Computation

A deterministic CPU inside neural layers raises security questions that conventional language-model deployment discussions can miss.

The first question is **where computation can be observed**.

If an evaluation or monitoring system inspects only ordinary assistant messages, it may not see the full sequence of intermediate state changes in recurrent layers.

The author explores this broader concern in follow-up work on hybrid models hiding information in recurrent state and on embedding oversight mechanisms into compiled weights. These are experimental demonstrations, not evidence that every deployed hybrid model is secretly running unmonitored programs. [11][12]

A second question is **what execution authority the embedded program actually has**.

A RISC-V instruction that loads from address `0x100` does not mean the model can load from the host process's address `0x100`.

In Qwenputer, the code operates within a simulated address space represented by recurrent state. There is no ordinary host operating system inside that address space, and the processor is not automatically given arbitrary filesystem, network, or process privileges.

The absence of an external tool call does not automatically make the system safer, however.

An untrusted program can still consume significant computation, loop indefinitely, corrupt its own simulated state, or exploit mistakes in the bridge or host-side orchestration if those components expose privileged operations.

A third issue is **state isolation**.

If a serving system supports multiple users, it must ensure that one user's CPU registers, RAM, firmware state, or filesystem snapshot are never reused as another user's state without proper isolation and initialization.

This follows from the architecture itself. Recurrent state is mutable data. It should be treated as tenant-sensitive state, not as an innocuous implementation detail.

A fourth issue is **checkpoint provenance**.

A pretrained model with additional hand-compiled layers is executable software in a meaningful sense. Those layers embody specified functions. An operator evaluating a checkpoint should inspect not just its model card and natural-language behavior, but also any attached circuits, special tokens, state initialization code, and external bridge components.

The general safeguards resemble those used for conventional code execution: explicit permission boundaries, resource budgets, independent validation, reproducible builds, state isolation, monitoring at the actual execution layer, and skepticism toward unverified model artifacts.

None of this proves that the Qwenputer checkpoint is malicious. It shows why the ability to compile computation into weights deserves serious systems and security analysis.

## 22. Unanswered Research Questions and Useful Extensions

Qwenputer opens several productive research directions.

**Can the compiler be made smaller and more general?** The current system targets a carefully constrained Qwen3.5 architecture. Supporting other hybrid recurrent models would require compatible primitives and precise numerical analysis.

**Can the same architecture execute more efficiently?** A specialized executor already improves throughput. Fused kernels or more direct state-machine execution may reduce overhead further, although doing so can erode the distinction between in-weight execution and a purpose-built external interpreter.

**How portable is numerical exactness?** Different GPUs, attention kernels, precision settings, quantization schemes, and normalization implementations can change numerical behavior. A robust system should validate those paths independently.

**Can learned and compiled components interact without losing guarantees?** The language model may choose which computation to perform while the compiled circuit enforces exact behavior during the computation. That separation is powerful, but the interface remains a source of agent-level errors.

**Can recurrent memory be independently audited?** A system that records state transitions, checks ownership, and compares trace hashes against a reference machine could make hidden computation more observable.

**Can compiled weights support trustworthy monitors?** If certain checks are imposed by circuits the language model cannot directly rewrite, they might provide a different kind of oversight mechanism. Any such claim would need a rigorous threat model and adversarial evaluation.

**What is the right boundary between an embedded circuit and an external tool?** Compiling a CPU into weights demonstrates possibility. It does not establish that this is preferable for production calculators. Comparing end-to-end reliability, isolation, transparency, compute cost, and latency will help answer where the design actually belongs.

A useful next project would be to extend our educational interpreter with more RV32I operations, build a reference execution engine independent of the PyTorch components, and generate randomized short programs. For each program, compare registers and memory after every instruction. Only after that works should we attempt to replace the Python decoder with fixed-weight circuit logic.

## 23. Conclusion: The Boundary Between a Model and a Machine Is Becoming More Interesting

Qwenputer is compelling because it inverts a familiar assumption.

We usually think of a language model as a learned function running *on* a computer.

Qwenputer places a deterministic computer *inside* a language model's forward computation.

It does so by choosing a recurrent architecture with a writable state matrix, making that state addressable through one-hot keys, representing values with carefully controlled numerical semantics, and compiling CPU logic into neural-network-compatible operations.

The achievement is not that Qwen has suddenly learned flawless arithmetic.

The CPU is designed, compiled, and checked. The model learns neither the instruction set nor its arithmetic semantics during this experiment.

The achievement is that the same architecture that normally processes language can also serve as a substrate for deterministic stateful computation.

Our educational experiment demonstrates the core principles on a much smaller scale. One-hot delta updates implement addressable memory. Fixed neural weights implement exact bitwise gates. A small RV32 instruction subset operates over that memory, and randomized tests confirm that its restricted arithmetic agrees with a reference implementation.

The full Qwenputer project goes further by compiling instruction execution, memory operations, and output control into a pretrained model's actual layer structure.

For machine learning researchers, it connects linear attention and fast-weight memory to program execution.

For compiler engineers, it offers an unusual target architecture where tensors, nonlinearities, and recurrent state replace conventional gates and registers.

For security researchers, it challenges the assumption that visible model text necessarily reveals all computation taking place during inference.

And for anyone building agents, it reinforces a familiar but important distinction: **exact computation and correct decision-making are not the same thing**.

A deterministic machine can produce a flawless intermediate result. The language model using that result can still make a mistake.

That is not a contradiction. It is a reminder that hybrid intelligence systems need to be evaluated at every boundary between perception, control, execution, and communication.

---

## 24. References and Source Code

**[1] Divyansh Agrawal.** *Qwenputer: a fully functioning, deterministic RISC-V CPU inside Qwen3.5.* October 22, 2026. Original project explanation, architecture overview, benchmarks, experimental filesystem, and limitations.

https://divagr.com/blog/qwenputer

**[2] Divyansh Agrawal.** *Qwenputer GitHub repository.* Source code, README, evaluation scripts, and example model commands.

https://github.com/divagr18/qwenputer

**[3] Qwenputer developers.** *RV32 inside Qwen3.5.* Detailed RISC-V design notes, instruction encodings, memory layout, compliance checks, execution measurements, firmware integration, and compiler stages.

https://github.com/divagr18/qwenputer/blob/main/experimental/riscv/docs/rv32.md

**[4] Qwenputer source code.** `chipformer/q35/ir.py`. Intermediate representation for circuit scheduling and Qwen3.5-compatible recurrent and feed-forward operations.

https://github.com/divagr18/qwenputer/blob/main/chipformer/q35/ir.py

**[5] Qwenputer source code.** `chipformer/q35/emit.py`. Translation from scheduled circuit graphs into model weights and residual layouts.

https://github.com/divagr18/qwenputer/blob/main/chipformer/q35/emit.py

**[6] Qwenputer source code.** `chipformer/q35/prims.py`. Logic and arithmetic compilation primitives.

https://github.com/divagr18/qwenputer/blob/main/chipformer/q35/prims.py

**[7] Songlin Yang, Jan Kautz, and Ali Hatamizadeh.** *Gated Delta Networks: Improving Mamba2 with Delta Rule.* ICLR 2025.

https://arxiv.org/abs/2412.06464

**[8] Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber.** *Linear Transformers Are Secretly Fast Weight Programmers.* ICML 2021.

https://arxiv.org/abs/2102.11174

**[9] Qwen.** *Qwen3.5-4B model card.* Base-model architecture and usage documentation.

https://huggingface.co/Qwen/Qwen3.5-4B

**[10] RISC-V International.** *RISC-V Instruction Set Manual.* Specification for RV32I, integer multiply/divide, and related extensions.

https://github.com/riscv/riscv-isa-manual

**[11] Divyansh Agrawal.** *Hybrid models can hide reasoning in their RNN state.* Follow-up experimental research on recurrent-state observability.

https://divagr.com/blog/hybrid-models-can-hide-reasoning-in-their-rnn-state

**[12] Divyansh Agrawal.** *Building oversight tools into a model's weights.* Follow-up experimental research on compiled oversight circuits.

https://divagr.com/blog/building-oversight-tools-into-a-models-weights

**[13] Qwenputer model checkpoint.** Hugging Face distribution of the modified Qwen3.5-4B checkpoint.

https://huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer

**Reproducibility and scope note, October 30, 2026:** The Qwenputer implementation details and published accuracy figures are drawn from the original author and repository. I independently implemented and ran a small CPU-only educational lab with DeltaNet-style one-hot memory, fixed-weight logic gates, and a Python-decoded subset of RV32I. Ten unit tests passed. A three-seed benchmark reported zero mismatches across the tested operations. I did not download or execute the full 18.7 GB Qwenputer checkpoint, reproduce its GPU traces, or independently verify its full ISA-compliance claims. The educational lab is not an official implementation, an exact numerical reproduction of Qwen3.5 kernels, or a substitute for the original compiler.
