# physcons

**Status:** Toy proof-of-concept. Not production-ready. Zero users. Not submitted to any venue.

## What is this?

A minimal implementation combining ideas from three papers into a toy 1D conservation demo:

- **Shodh-MoE** ([arXiv:2605.15179](https://arxiv.org/abs/2605.15179)) — patch-based routing
- **PI-HC-MoE** ([arXiv:2402.13412](https://arxiv.org/abs/2402.13412), ICLR 2024) — hard conservation constraints via MoE
- **PI-MFM** ([arXiv:2512.23056](https://arxiv.org/abs/2512.23056)) — physics-informed residual correction

**What this does:**
- Routes patches of a 1D scalar field through tiny "expert" networks (simple MLPs)
- Computes conservation residuals using a Rust kernel (with Python fallback)
- Applies hard corrections when the conserved quantity drifts beyond a threshold

**What this is NOT:**
- Not a multimodal foundation model
- Not a 2D/3D PDE solver
- Not a molecular dynamics engine
- Not a reproduction of the cited papers
- Not a Claude/GPT/Gemini physics benchmark
- Not validated on real data
- Not submitted to NeurIPS/ICLR/CVPR

## Installation

Requires Python 3.8+ and Rust toolchain (for building the Rust extension).

### Install Rust (if not already installed)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Build and install

```bash
pip install maturin
maturin develop  # For development
# OR
maturin build --release  # For release build
pip install target/wheels/physcons-*.whl
```

Alternatively, for editable install:

```bash
pip install -e .
```

## Usage

After installation, run the demo:

```bash
physcons demo
```

This will:
1. Generate a conserved 1D field
2. Generate a broken (non-conserved) field
3. Route both through the patch router
4. Show residuals before/after hard correction

**No performance metrics are recorded.** If you want to measure anything, modify the code yourself.

## Architecture

```
1D Field → Patch Router → Conservation Check → Hard Correction (if needed)
              (Python)       (Rust kernel)         (Rust kernel)
```

- **Data generation** (`physcons/data.py`): Synthetic 1D fields
- **Patch router** (`physcons/router.py`): Splits field into patches, routes to experts
- **Conservation layer** (`physcons/conservation.py`): Computes residuals, applies corrections
- **Rust kernel** (`physcons_rust/src/lib.rs`): Fast residual computation with fail-safes

## Tests

Tests are written but execution is left to you:

```bash
pip install pytest
pytest tests/
```

## Limitations

- **Toy 1D only**: Not 2D/3D, no real PDEs
- **Simple experts**: Linear layers, not foundation models
- **No LLM integration**: Could swap in API calls later, but doesn't now
- **Deterministic routing**: Based on patch mean, not learned
- **No real physics**: Synthetic data, not validated against experiments
- **Zero deployments**: Untested in production

## Papers (framing only)

We cite these papers for problem framing. We do NOT reproduce their methods or vendor their code:

- **PI-MFM**: Residual-based physics correction in multimodal FMs  
  https://arxiv.org/abs/2512.23056

- **PI-HC-MoE**: Hard conservation constraints via mixture of experts  
  https://arxiv.org/abs/2402.13412 (ICLR 2024)

- **Shodh-MoE**: Patch-based routing for efficient inference  
  https://arxiv.org/abs/2605.15179

## Future (maybe)

If someone wanted to extend this (not claiming we will):
- Swap simple MLPs for actual transformer experts
- Add LLM API calls (OpenAI, Anthropic, etc.) as "oracle experts"
- Scale to 2D/3D fields
- Real PDE integration (FEM, FVM, etc.)
- Benchmark on physics datasets (OpenFOAM, materials science, etc.)

But right now: none of that exists.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Disclaimer

This is a toy PoC. Do not use for production. Do not cite as a reproduced method. Do not claim acceptance at any conference. Honestly: it's a demo to explore an idea.
