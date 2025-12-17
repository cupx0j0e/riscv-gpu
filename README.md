# RISC-V GPU Research Repository

This repository is a research and learning workspace for building a RISC-V based GPU architecture that avoids custom ISA extensions (the primary exception being the RISC-V Vector extension where appropriate). The goal is to explore GPU building blocks, accelerator micro-architectures, and end-to-end flows using open tools and standard RISC-V features.

Status: initial work focused on systolic-array based matrix-multiply building blocks (2x2 and 4x4 example designs), RTL testbenches, and simple simulation flows.

Contents
- **Overview**: goals and design philosophy.
- **systolic_array/**: starting point — example systolic array RTL for matmul and simulation flows.
- **Roadmap**: planned next steps for the GPU stack.

Why this repo
----------------
GPUs are complex systems built from well-understood building blocks: vector/SIMD units, memory hierarchies, data-movement engines, and computation accelerators. This repo aims to:

- Study those building blocks in isolation and in combination.
- Build hardware blocks using synthesizable RTL and open-source tools.
- Keep the ISA standard (RISC-V + Vector extension) to maximize portability.

What you'll find here
----------------------
Top-level structure (high-level):

- `systolic_array/` — Systolic-array examples and simulation flows.
	- `2x2_matmul/` — 2x2 processing-element (PE) matrix multiply example.
		- `rtl/` — Verilog sources for the PE and top-level systolic array.
			- [systolic_array/2x2_matmul/rtl/mac_pe.v](systolic_array/2x2_matmul/rtl/mac_pe.v)
			- [systolic_array/2x2_matmul/rtl/systolic_array.v](systolic_array/2x2_matmul/rtl/systolic_array.v)
			- [systolic_array/2x2_matmul/rtl/systolic_array_tb.v](systolic_array/2x2_matmul/rtl/systolic_array_tb.v)
		- `simulation/` — Makefile and helper files to run iverilog/vvp simulations and produce waveforms.
			- [systolic_array/2x2_matmul/simulation/Makefile](systolic_array/2x2_matmul/simulation/Makefile)
	- `4x4_matmul/` — 4x4 PE example and testbench.
		- [systolic_array/4x4_matmul/rtl/pe_4x4.v](systolic_array/4x4_matmul/rtl/pe_4x4.v)
		- [systolic_array/4x4_matmul/rtl/systolic_array_4x4.v](systolic_array/4x4_matmul/rtl/systolic_array_4x4.v)
		- [systolic_array/4x4_matmul/simulation/Makefile](systolic_array/4x4_matmul/simulation/Makefile)

Getting started
-----------------
Prerequisites (recommended):

- `iverilog` + `vvp` (simulation)
- `yosys` (synthesis / linting)
- `gtkwave` or any VCD viewer (optional)

On macOS (Homebrew):

```bash
brew install icarus-verilog yosys gtkwave
```

Running the provided simulations
---------------------------------
Each example has a `simulation/` directory with a `Makefile` to run the RTL tests and generate waveforms. Example:

```bash
# 2x2 example
cd systolic_array/2x2_matmul/simulation
make

# 4x4 example
cd ../../4x4_matmul/simulation
make
```

The Makefiles typically run `iverilog` to produce a `sim.vvp` binary and then run `vvp sim.vvp` producing `wave.vcd`; open that VCD with `gtkwave wave.vcd`.

Design notes
-------------
- Systolic arrays are used here as a compute substrate for dense matrix multiply (GEMM). They demonstrate dataflow-style mapping of compute to constant-mesh PE arrays and are a common accelerator building block in GPUs.
- The long-term plan is to integrate these accelerators with a RISC-V CPU and expose work to them using standard mechanisms (memory-mapped queues, DMA engines, or offload semantics) while keeping the ISA itself standard-compliant.
- The RISC-V Vector extension is considered the primary ISA-level mechanism for expressing wide-data parallelism; accelerators (like systolic arrays) are complementary and can be targeted from vectorized code or from higher-level runtimes.

Roadmap: Building the RISC-V GPU
----------------------------------

The goal is to build a GPU where the compute cores execute standard RISC-V instructions (RV32IMV or RV64GV), avoiding custom ISA extensions beyond the Vector extension.

### Architecture Overview

A GPU consists of:
1. **Many parallel execution units** — SIMT lanes organized into warps
2. **A warp/thread scheduler** — manages thousands of concurrent threads
3. **Memory hierarchy** — registers, shared memory, caches, global memory
4. **Command frontend** — receives work from host, dispatches to compute units

### Phase 1: Single RISC-V Vector Core
- [ ] Integrate or build a minimal RV32IM core (candidates: PicoRV32, VexRiscv, or custom)
- [ ] Add Vector extension support (study Ara, Vicuna, or build minimal vector ALU)
- [ ] Test with simple vector programs: vector add, dot product, small matmul
- [ ] Establish simulation and verification flow

### Phase 2: Multi-lane SIMT Execution
- [ ] Instantiate multiple scalar pipelines sharing instruction fetch (warp)
- [ ] Build warp scheduler (round-robin or scoreboard-based)
- [ ] Implement divergence handling (masked execution for branches)
- [ ] Add warp-level synchronization primitives

### Phase 3: Memory Hierarchy
- [ ] Design banked register file (GPUs need large register files)
- [ ] Add shared memory / scratchpad (fast, software-managed, per-workgroup)
- [ ] Implement L1 data cache or texture-cache style access
- [ ] Build global memory interface (AXI or similar to external DRAM)

### Phase 4: Compute Dispatch & Host Interface
- [ ] Command processor: read work descriptors from memory
- [ ] Thread block / workgroup dispatcher: assign blocks to compute units
- [ ] Barrier / sync support (`__syncthreads()` equivalent)
- [ ] DMA engine for bulk data movement

### Phase 5: Software Stack
- [ ] Minimal runtime to launch kernels from host
- [ ] Use LLVM RISC-V backend with vector intrinsics for compilation
- [ ] Write example GPU kernels in C with RVV intrinsics
- [ ] Simple benchmarks: SAXPY, GEMM, reduction

### Reference Projects

| Project | Description |
|---------|-------------|
| [Vortex](https://github.com/vortexgpgpu/vortex) | Open-source RISC-V GPGPU — closest reference architecture |
| [Ara](https://github.com/pulp-platform/ara) | Full RVV 1.0 vector unit from ETH Zurich |
| [Vicuna](https://github.com/vproc/vicuna) | Lightweight RVV core |
| [VexRiscv](https://github.com/SpinalHDL/VexRiscv) | Configurable RISC-V in SpinalHDL |
| [PicoRV32](https://github.com/YosysHQ/picorv32) | Tiny RV32 core, easy to understand |

### Current Status

✅ **Done**: Systolic array building blocks (2×2, 4×4 matmul PEs) — understanding dataflow compute  
🔄 **Next**: Phase 1 — integrate a base RISC-V core and add vector support

Contributing
-------------
Contributions are welcome. Suggested workflow:

1. Open an issue describing the change or feature.
2. Create a branch and send a PR with clear description and tests where applicable.

Licensing
---------
This repository does not include an explicit license file yet. If you want to add a license, consider a permissive license such as MIT or Apache-2.0.

Contact / Questions
--------------------
Open an issue or create a discussion in this repository for questions, design discussions, or coordination.

Acknowledgements
-----------------
This work uses open-source tools such as Icarus Verilog and Yosys for simulation and synthesis research.

--
