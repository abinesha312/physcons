# Development Guide

## Quick Start

### Prerequisites

- Python 3.8+
- Rust toolchain (install via [rustup](https://rustup.rs/))
- pip

### Development Setup

1. **Clone the repo** (already done if you're reading this)

2. **Install Rust** (if needed):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env
   ```

3. **Install maturin** (Python-Rust build tool):
   ```bash
   pip install maturin
   ```

4. **Build the Rust extension in development mode**:
   ```bash
   maturin develop
   ```
   
   This compiles the Rust code and installs the package in editable mode.

5. **Verify installation**:
   ```bash
   python -c "import physcons; print(physcons.__version__)"
   physcons demo
   ```

### Running Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v
```

### Building for Distribution

Release build (optimized):
```bash
maturin build --release
```

Wheels will be in `target/wheels/`.

### Code Structure

```
physcons/
├── src/physcons/          # Python package
│   ├── __init__.py
│   ├── cli.py             # CLI entry point
│   ├── data.py            # Synthetic field generation
│   ├── router.py          # Patch routing logic
│   └── conservation.py    # Conservation layer (calls Rust)
├── physcons_rust/         # Rust extension
│   ├── Cargo.toml
│   └── src/lib.rs         # Conservation kernel
└── tests/                 # pytest tests
```

### Modifying the Code

- **Python changes**: Just edit and run (editable install via `maturin develop`)
- **Rust changes**: Re-run `maturin develop` after editing `.rs` files
- **Tests**: Add to `tests/test_*.py`, run with `pytest`

### Troubleshooting

**"Rust kernel not available"**: 
- Run `maturin develop` to build the Rust extension
- Package will fall back to pure Python (slower but works)

**Import errors**:
- Make sure you're in the project directory
- Check that `maturin develop` completed successfully

**Build errors**:
- Ensure Rust toolchain is installed: `rustc --version`
- Update maturin: `pip install -U maturin`
