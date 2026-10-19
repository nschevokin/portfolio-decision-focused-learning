# Portfolio Allocation with Decision-Focused Learning

A neural portfolio allocator trained **end-to-end through a convex optimisation layer** (decision-focused learning, DFL) and compared against classical Mean-Variance Optimisation (MVO) on a fully held-out, walk-forward test period.

**Author:** Nikita Chevokin 

**Stack:** PyTorch · cvxpy / cvxpylayers · Optuna · scikit-learn · statsmodels · yfinance · pandas

---

## The idea

Classical pipelines predict returns first and optimise afterwards, so the predictor is never trained for what the portfolio actually needs. Here the optimiser sits **inside** the network: the loss is computed on realised portfolio performance and its gradient flows back through the quadratic program into the predictor.

```
5-day window × 6 features per stock
  → GRU (per stock, over time)
  → Cross-sectional multi-head attention (across the 50 stocks)
  → expected-return estimates μ̂
  → differentiable QP layer (cvxpylayers): long-only, weights sum to 1, 15% cap, risk term
  → portfolio weights
  → loss = −(mean return / EMA std) + turnover penalty + CVaR tail penalty
```

The risk term uses a Cholesky parametrisation (‖Lᵀw‖² = wᵀΣw) so the problem stays DPP-compliant and gradients through the layer are valid.

## Data and setup

- **Universe:** 50 US large-cap stocks, adjusted close from Yahoo Finance, 2013-01-02 → 2026-02-27 (3,309 trading days, no missing values).
- **Features (6):** `mom_5d`, `mom_21d`, `down_vol`, `risk_mom`, `price_rank_52w`, `rsi`, converted to cross-sectional ranks, winsorised and Yeo-Johnson transformed (fitted on train only). Prices are **fractionally differenced** (d = 0.5, the minimum passing ADF on train) to keep memory while being stationary.
- **Target:** 5-day forward return.
- **Splits (chronological, 5-day purge at each boundary):**

| Split | Period | Days |
|---|---|---|
| Train | 2013-12 → 2022-06 | 2,132 |
| Validation | 2022-07 → 2024-04 | 448 |
| Test | 2024-05 → 2026-02 | 453 |

- **Covariance:** 63-day rolling sample covariance for DFL; 252-day Ledoit-Wolf for MVO.
- **Costs:** flat 10 bps per unit of one-way turnover.
- **Tuning:** 20 Optuna trials on validation Sharpe (final: hidden 128, lr 3.9e-4, dropout 0.13, CVaR weight 0.093).
- **Walk-forward:** the canonical DFL series is retrained every ~6 months on an expanding window (3 segments) and the out-of-sample slices are concatenated.

## Baselines

- **MVO (LW μ):** Ledoit-Wolf shrinkage on Σ, sample-mean μ.
- **MVO (JS μ):** adds James-Stein shrinkage on μ, the stronger and more realistic baseline.
- **EW:** equal-weight.

## Results (test set, 448 days)

| Strategy | Ann. return | Ann. vol | Sharpe | Max drawdown |
|---|---|---|---|---|
| **DFL (walk-forward)** | 123.7% | 35.9% | **3.44** | −39.8% |
| DFL (single train) | 82.3% | 37.6% | 2.19 | −59.8% |
| DFL (top-K, constrained) | 54.6% | 37.5% | 1.46 | −66.8% |
| MVO (LW μ) | 23.6% | 24.1% | 0.98 | −25.4% |
| MVO (JS μ) | 23.6% | 24.1% | 0.98 | −25.4% |
| Equal-weight | 20.5% | 15.2% | 1.35 | −18.4% |

**Statistical checks (DFL walk-forward vs MVO JS μ):** paired t-test on daily differences p = 0.002; stationary block-bootstrap 95% CI for the Sharpe difference [+0.60, +4.72]; Deflated Sharpe Ratio (corrects for the 20 tuning trials) reports the result as robust.

**Sub-window Sharpe (walk-forward DFL):** 5.71 → 3.31 → 1.45 across the three test windows, so the headline is dominated by the first window.

**Ablation (validation Sharpe, same QP layer and budget):** GRU + attention 3.76 · linear 3.49 · LSTM 3.16. Most of the gain comes from the DFL layer itself; cross-sectional attention adds a smaller increment.

### Top-K constrained variant

A second model follows the original brief more literally: only the top-K predicted names may receive weight (cardinality mask inside the QP) and risk is a **hard variance cap** (30% annualised) instead of a soft penalty. It holds about 7 names per day. Against MVO (JS μ) and equal-weight the difference is **not statistically significant** (p = 0.35 and 0.27), and its drawdown is the largest of the group.

## What mattered in practice

- **Look-ahead removal:** wavelet denoising was dropped because the inverse transform is non-causal; fractional differencing is causal.
- **Tail risk:** the first version reached Sharpe ≈ 1.37 with a −67% drawdown; adding a differentiable **CVaR** term to the loss addressed tail losses.
- **Chronological batches:** `shuffle=False`, so the carried-over previous weights and turnover penalty reflect a real lagged allocation.

## Limitations (as documented in the notebook)

- **Survivorship bias:** the 50 names were picked in 2026 from currently listed companies, which biases every long-only result upward, DFL and baselines alike.
- **Tech tilt and regime:** 32% of the universe is Tech and the test window coincides with a strong Tech-led rally; momentum has not been tested through a regime like 2022.
- **Window dominance:** the first test window drives the headline Sharpe.
- **Single seed**, stylised transaction costs, and historical (not forward-looking) covariance.
- **Selection:** 20 Optuna configurations were tried and the best reported; the Deflated Sharpe Ratio is used to account for this.

## Repository

```
portfolio-decision-focused-learning/
├── README.md
├── notebook.ipynb      # sections: data, quality, EDA, features, splits, MVO, DFL, ablation, evaluation
└── assets/             # key figures exported from the notebook
```

## Running it

Developed in Google Colab. The first cell installs `cvxpylayers cvxpy PyWavelets optuna yfinance statsmodels`. Run top to bottom; the walk-forward retraining of the top-K variant is slow (about 53 minutes in the saved run).

## License

MIT, see [LICENSE](./LICENSE).
