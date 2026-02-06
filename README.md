# Two State Ontology (TSO) — Quantum Validation

**Please try to falsify this.**

TSO derives quantum phase transition behavior from percolation theory. The predictions come from mathematics *before* comparison with experiment — no parameter fitting.

## Results

| Platform | Tests | Passed | Notes |
|----------|-------|--------|-------|
| IBM Marrakesh | 12 | 11 | Failed test was programming error (fixed) |

## Core Predictions

TSO **derives** these values (not fitted):

| Parameter | Value | Source |
|-----------|-------|--------|
| p_c | 0.3116 | 3D site percolation threshold |
| N_c | 6 | Cubic lattice coordination number |
| α | 7/(1 + e×ρ_c) ≈ 2.58 | Hilbert space geometry |

### The Key Claim

At the measurement-induced phase transition (MIPT):

```
S(p) = S_max × tanh(κ × N_c × (p_c - p) / p_c)
```

The transition occurs at **p_c ≈ 0.31** with sigmoid crossover.

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR-USERNAME/tso-validation.git
cd tso-validation

# Install
pip install -r requirements.txt

# Run on simulator
python tso_validation.py --backend aer_simulator

# Run on IBM hardware (requires IBM Quantum account)
python tso_validation.py --backend ibm_marrakesh --shots 4096
```

## How to Falsify TSO

| Prediction | How to Break It |
|------------|-----------------|
| p_c = 0.31 ± 0.02 | Find transition at different threshold |
| Sigmoid crossover | Show linear or step-function transition |
| κ varies with topology | Show κ constant across all connectivities |

## Files

```
tso-validation/
├── README.md              # This file
├── tso_validation.py      # Main validation script
├── requirements.txt       # Dependencies
├── LICENSE               # MIT License
└── results/              # Validation data
```

## Author

John Pepin — Independent Researcher — [incapp.org](https://incapp.org)

## License

MIT License — Use freely, attribute appropriately.
