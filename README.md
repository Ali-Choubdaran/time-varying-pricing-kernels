# Inferring Subjective Probabilities from Option Prices via Time-Varying Pricing Kernels

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-research-orange.svg)

**Author:** Ali Choubdaran  
**Institution:** London School of Economics  
**Date:** 2024

## Abstract

This repository implements a novel methodology for estimating time-varying stochastic discount functions (SDFs) where pricing kernel coefficients depend on risk-neutral moments. My approach resolves the long-standing monotonicity puzzle in asset pricing while significantly improving return prediction accuracy through better inference of subjective probabilities from option prices.

## Table of Contents

- [Background and Motivation](#background-and-motivation)
- [The Monotonicity Puzzle](#the-monotonicity-puzzle)
- [My Innovation](#my-innovation)
- [Mathematical Framework](#mathematical-framework)
- [Implementation](#implementation)
- [Key Results](#key-results)
- [Installation and Usage](#installation-and-usage)
- [Repository Structure](#repository-structure)
- [Citation](#citation)

## Background and Motivation

### The Traditional Approach

In asset pricing, the fundamental relationship between risk-neutral and subjective (physical) probabilities is:

```
p(r) = q(r) / SDF(r)
```

where:
- `p(r)` = subjective probability of return r
- `q(r)` = risk-neutral probability from option prices
- `SDF(r)` = stochastic discount function (pricing kernel)

Traditional approaches estimate **time-invariant** pricing kernels, treating the SDF as constant across different market conditions. This "one-size-fits-all" approach suffers from several critical limitations:

1. **Information Waste**: Uses only a single point (realized return) while ignoring the entire risk-neutral distribution
2. **Monotonicity Puzzle**: Produces pricing kernels that increase at high returns, violating economic theory
3. **Poor Prediction**: Fails to capture time-varying risk aversion and market sentiment

### The Core Problem

The issue lies in **locality**: traditional methods assume that `p(r)` depends only on `q(r)` and `SDF(r)` at that specific return level. However, this ignores how the **entire shape** of the risk-neutral distribution affects pricing and probability assessment.

Consider this: if you observe a risk-neutral distribution with high implied volatility versus one with low volatility, shouldn't this affect how you interpret probabilities across all return levels? The traditional approach says no – my approach says yes.

## The Monotonicity Puzzle

### What is the Puzzle?

Economic theory predicts that pricing kernels should be **monotonically decreasing**: higher returns correspond to states where marginal utility is lower, so the pricing kernel should be smaller. However, empirical studies consistently find that unconditional pricing kernels **increase** at high returns – a puzzle that has persisted for decades.

### My Explanation

The puzzle arises because researchers estimate **unconditional** (time-averaged) pricing kernels when the true kernels are **time-varying**. Here's the intuition:

1. **Individual Kernels**: Each `SDF_t(r)` at time t is monotonically decreasing
2. **Shape Variation**: In high-volatility periods, the kernel becomes flatter
3. **Sampling Bias**: High-volatility periods are more likely to produce extreme returns
4. **Aggregation Effect**: When averaged, the flatter curves (sampled at extreme points) create an apparent upturn

Think of it as a statistical artifact: you're seeing the average of many well-behaved functions, but the averaging process creates the illusion of misbehavior.

## My Innovation

### Two Key Contributions

#### 1. Resolving the Monotonicity Puzzle

I estimate **time-varying** pricing kernels where each individual `SDF_t(r)` is globally decreasing, but their time variation creates the observed non-monotonicity in unconditional averages. My results show:

- Individual pricing kernels are monotonic in each time period
- The puzzle disappears when time variation is properly modeled
- Risk aversion co-moves strongly with market volatility

#### 2. Superior Return Prediction

By properly accounting for time-variation, I achieve:
- **5.18% out-of-sample R²** vs 2.72% for benchmark models
- **50% improvement in Sharpe ratio** for market timing strategies
- More accurate inference of subjective probabilities

### The Non-Local Insight

My key innovation is making the pricing kernel depend on **risk-neutral moments** rather than just the local return value. This creates a **non-local** relationship where the entire risk-neutral distribution affects probabilities at each point.

Instead of:
```
p(r) depends only on q(r) and SDF(r) locally
```

I have:
```
p(r) depends on q(r) and SDF(r, moments_of_q(.))
```

This captures how market participants' risk assessment is influenced by the overall shape of implied distributions, not just point estimates.

## Mathematical Framework

### Model Specification

My SDF takes the form:

```
SDF(r,t) = exp(a₀,t + a₁,t·r + a₂,t·r² + a₃,t·r³)
```

Where each coefficient is a polynomial in risk-neutral moments:

```
aᵢ,t = bᵢ₀ + bᵢ₁·m₁,t + bᵢ₂·m₂,t + ... + polynomial terms
```

with `mₖ,t` being the k-th standardized risk-neutral moment at time t.

### Key Constraint

The intercept `a₀,t` is **not estimated** but determined by the normalization constraint:

```
∑ᵣ p(r,t) = 1
```

This reduces the parameter space and ensures proper probability distributions.

### Numerical Implementation

To prevent overflow in the exponential function, I use log-space arithmetic:

1. Compute the "free" part: `linear_part = a₁,t·r + a₂,t·r² + a₃,t·r³`
2. Apply log-space normalization to find `a₀,t`
3. Return `log(SDF) = a₀,t + linear_part`

This approach is numerically stable even for extreme parameter values encountered during optimization.

### Estimation Method

I use **maximum likelihood estimation** where the likelihood of observing return `rₜ` is the subjective probability `p(rₜ)`. The objective function is:

```
L = ∑ₜ log(SDF(rₜ,t))
```

since `log(p(rₜ)) = log(q(rₜ)) - log(SDF(rₜ,t))` and `log(q(rₜ))` is constant across parameter values.

## Implementation

### System Requirements

- Python 3.8+
- NumPy, Pandas, SciPy
- 8GB+ RAM (for full dataset)
- Multi-core CPU recommended for optimization

### Quick Start

```python
from option_pricing import Config, run_estimation, run_optimization

# Configure model
config = Config(
    min_return=0.8,        # -20% to +20% return range
    max_return=1.2,
    n_grid_points=401,     # Match data structure
    sdf_degree=3,          # Cubic polynomial
    coef_degree=2,         # Quadratic moment dependence
    moment_orders=[2, 3]   # Variance and skewness
)

# Run estimation
results = run_estimation("Q_ready_401_detailed.dta", config)

# Optimize parameters
final_results = run_optimization(results, n_starts=3, param_bounds=2.0)

print(f"Best likelihood: {final_results['optimization']['best_value']:.2f}")
```

### Data Requirements

The input data (`Q_ready_401_detailed.dta`) contains:
- `date`: Trading dates
- `mr`: Realized market returns (simple returns)
- `ar800` to `ar1200`: Arrow-Debreu prices for returns -20% to +20%
- `first_ar`, `last_ar`: Coverage indicators for data quality

## Key Results

### Monotonicity Resolution

My approach shows that:
- Individual time-varying pricing kernels are **globally monotonic**
- The puzzle arises from **time-aggregation effects**
- Risk aversion varies significantly with market volatility
- High-volatility periods have flatter (but still decreasing) pricing kernels

### Prediction Performance

| Model | α | β | R² (%) | R²_OS (%) |
|-------|---|---|--------|-----------|
| **Time-Varying SDF** | -0.003 | 1.909 | 8.3 | **5.18** |
| SVIX² Benchmark | -0.004 | 2.077 | 5.6 | 2.72 |

### Market Timing Strategy

Using inferred subjective probabilities for mean-variance optimization:
- **S&P 500 Sharpe Ratio**: 4.4%
- **My Strategy Sharpe Ratio**: 6.6% (**50% improvement**)
- Average equity allocation: 56%

The strategy dynamically allocates between stocks and Treasuries based on the inferred expected return-to-volatility ratio from subjective probabilities.

### Risk Aversion Dynamics

My estimated risk aversion parameter:
- **Mean**: 2.84
- **Standard Deviation**: 0.79
- **Range**: 0.25 to 6.30
- **Strong correlation with VIX**: Captures market stress periods

## Repository Structure

```
├── option_pricing.py          # Main implementation
├── Q_ready_401_detailed.dta   # Arrow-Debreu price data
├── requirements.txt           # Python dependencies  
├── README.md                  # Documentation
└── LICENSE                    # MIT License
```

## Advanced Features

### Custom Moment Selection

```python
# Use only variance and skewness (faster estimation)
config = Config(moment_orders=[2, 3])

# Full moment set for maximum information
config = Config(moment_orders=[1, 2, 3, 4])
```

### Optimization Robustness

The implementation includes:
- **Multi-start optimization** to avoid local minima
- **Multiple algorithms** (L-BFGS-B, Nelder-Mead, Powell)
- **Automatic parameter bounds** to prevent numerical issues
- **Comprehensive logging** for diagnostics

### Model Diagnostics

```python
# Check estimation quality
results = run_estimation(data_path, config)
print(f"Observations: {len(results['dates'])}")
print(f"Moment correlations:\n{results['moments'].corr()}")

# Post-optimization analysis
opt_results = results['optimization']
if opt_results['success']:
    params = opt_results['best_params']
    # Analyze parameter estimates, compute SDF shapes, etc.
```

## Theoretical Extensions

### Connection to Behavioral Finance

My approach naturally incorporates behavioral elements:
- **Time-varying risk aversion** captures sentiment cycles
- **Non-local relationships** reflect how market participants process distributional information
- **Volatility clustering** in risk aversion matches observed patterns

### Relationship to Existing Literature

This work connects to several strands of research:
- **Option-implied information** (Bakshi et al., 2003; Bandi & Renò, 2012)
- **Pricing kernel estimation** (Aït-Sahalia & Lo, 2000; Jackwerth, 2000)
- **Return prediction** (Kelly & Pruitt, 2013; Martin & Wagner, 2019)
- **Time-varying risk premia** (Lettau & Ludvigson, 2001; Campbell & Thompson, 2008)

### Future Research Directions

Potential extensions include:
- **Multi-asset implementation** for cross-sectional pricing
- **High-frequency applications** using intraday option data
- **International markets** to test universality
- **Alternative utility specifications** beyond CRRA

## Installation and Usage

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/option-subjective-probabilities.git
cd option-subjective-probabilities

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import option_pricing; print('Installation successful')"
```

### Requirements

Create `requirements.txt`:
```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
statsmodels>=0.12.0
matplotlib>=3.4.0
```

### Basic Usage

```python
import option_pricing as op

# Load and run with defaults
results = op.run_estimation("Q_ready_401_detailed.dta")

# Optimize parameters
final_results = op.run_optimization(results)

# Extract key results
best_params = final_results['optimization']['best_params']
sdf_spec = final_results['sdf_spec']
```

### Advanced Configuration

```python
# Custom configuration for research
config = op.Config(
    min_return=0.7,        # Wider return range
    max_return=1.3,
    sdf_degree=4,          # Higher-order polynomial
    coef_degree=3,         # More complex moment dependence
    moment_orders=[1,2,3,4,5]  # Include fifth moment
)

results = op.run_estimation(data_path, config)
```

## Troubleshooting

### Common Issues

1. **Optimization fails to converge**
   - Reduce model complexity (lower polynomial degrees)
   - Increase parameter bounds or number of random starts
   - Check data quality (sufficient coverage, no extreme outliers)

2. **Numerical overflow warnings**
   - The log-space implementation should prevent this
   - If it persists, check for extreme parameter values
   - Consider tighter parameter bounds

3. **Poor prediction performance**
   - Verify data quality and date alignment
   - Try different moment combinations
   - Consider out-of-sample period selection

### Performance Optimization

- Use fewer moment orders for faster estimation
- Reduce grid resolution for preliminary testing
- Utilize multiple CPU cores for optimization
- Consider data subsampling for initial exploration

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{choubdaran2024subjective,
  title={Inferring Subjective Probabilities from Option Prices via Time-Varying Pricing Kernels},
  author={Choubdaran, Ali},
  year={2024},
  institution={London School of Economics},
  url={https://github.com/your-username/option-subjective-probabilities}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

**Ali Choubdaran**  
PhD Candidate, Finance  
London School of Economics  
Email: a.choubdaran-varnosfaderani@lse.ac.uk

For questions about the methodology, implementation, or data requirements, please open an issue on GitHub or contact me directly.

---

*This research contributes to understanding how market participants process option-implied information and form expectations about future returns. By properly accounting for time-varying risk preferences, I can better bridge the gap between risk-neutral and physical probabilities, ultimately improving both theoretical understanding and practical forecasting ability.*
