# Variational-EM k-selector — Step-1 (Basis A grounding) results

Config: k_max=6 (K=7 incl bypass), n=800, sigma=0.3, epochs=500, 5 seeds. Constant baseline (fixed mixture), bypass ON.

## Check 1 — surviving k vs mode spacing (resolvability)

| spacing (÷σ) | surviving k (median) | surviving k (mean) | bypass weight (mean) |
|---:|---:|---:|---:|
| 0.5 | 2 | 2.00 | 0.888 |
| 1.0 | 2 | 2.20 | 0.870 |
| 1.5 | 1 | 1.60 | 0.886 |
| 2.0 | 3 | 3.00 | 0.627 |
| 3.0 | 4 | 4.00 | 0.334 |
| 4.0 | 4 | 4.40 | 0.234 |
| 6.0 | 3 | 2.80 | 0.198 |

## Check 2 — α₀ insensitivity (resolved, separation=4)

| α₀ | surviving k (median) | surviving k (mean) |
|---:|---:|---:|
| 0.05 | 4 | 4.20 |
| 0.1 | 4 | 4.40 |
| 0.2 | 4 | 4.20 |
| 0.5 | 4 | 3.80 |
| 0.1429 | 4 | 4.20 |

## Check 3 — full shape vs summary (matched mean & variance)

Principled arbiter: held-out NLL of the genuine mixture vs a single Gaussian fit by moments (the summary model). Surviving-k is reported only as a caveated resolution diagnostic — it over-counts both shapes via bin tiling (FINDINGS.md).

| | bimodal | broad |
|---|---:|---:|
| surviving k (median, diagnostic) | 4 | 4 |
| empirical mean | 0.01 | -0.01 |
| empirical variance | 0.44 | 0.44 |
| held-out NLL — mixture | 0.921 | 1.072 |
| held-out NLL — single Gaussian | 1.018 | 1.008 |
| held-out NLL — bypass only | 6.742 | 22.722 |
| **mixture edge over summary (nats)** | **+0.096** | **-0.064** |

Matched mean & variance → a moment/summary objective sees them as identical; the genuine mixture's HELD-OUT NLL separates them (it helps on two peaks, not on the broad bell).

## Negative control — smooth unimodal (must not invent classes)

| noise σ | surviving k (median) | bypass weight (mean) |
|---:|---:|---:|
| 0.05 | 1 | 0.595 |
| 0.1 | 1 | 0.798 |
| 0.2 | 2 | 0.548 |
| 0.4 | 2 | 0.741 |
