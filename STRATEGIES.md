# STRATEGIES — Full Defense Catalog

## Overview

AEGIS v9.3 deploys **70 trap defenses** across 5 phases + 3 structural mechanisms + 2 security models. Every defense has a name, an origin metaphor, and a concrete algebraic operation.

---

## Phase 0: Entropy Collapse (3 mechanisms)

| # | Name | Operation | Purpose |
|---|------|-----------|---------|
| EC-1 | Column Flood | 8% of columns → PRF-random values | Destroy clean column structure |
| EC-2 | Block Permutation | 4-column blocks → GF(4) permutation shuffle | Break local patterns |
| EC-3 | Linear Combination | 50 target columns → sum of 8 random sources | Create algebraic dependencies |

## Phase I: 26 Ancient Traps

| # | Name | Origin | Operation |
|---|------|--------|-----------|
| T25 | Bronze Crossbows | Ancient warfare | 50 column pairs → linear offset |
| T24 | Murder Holes | Castle defense | High-weight columns → single flip |
| T23 | Dried Gut | Tripwire | Prime-index columns → perturbation |
| T22 | Portcullis | Gate defense | Columns 1165+ → 30% random flip |
| T21 | False Steps | Deception | 100 column swaps |
| T20 | Nightingale | Japanese floors | Cyclic stride perturbation |
| T19 | Pivot Doors | Hidden passages | Block-offset single flips |
| T18 | False Chambers | Honeypot | T_fake matrix deployed |
| T17 | Right-Hand | Bias exploitation | Upper 3 rows → 8% flip rate |
| T16 | Sand Pit | Collapse trap | 3+ changed columns → full randomize |
| T15 | Arrow-Tip | Geometric poison | Spread-line points added to columns |
| T14 | Siphons | Drainage | Triple-column linear combinations |
| T13 | Tide Wells | Gradient trap | Position-dependent flip probability |
| T12 | Suction Mud | Zero trap | 30 columns → all zeros |
| T11 | Haloclines | Duplication | 40 column duplications |
| T10 | Vibration | Pure noise | 20 columns → full random |
| T9 | Obsidian | Decoy T | T_decoy matrix deployed |
| T8 | Radon | Silent copy | 50 column copies |
| T7 | Spores | Local spread | Adjacent column recurrence |
| T6 | Forgetting | Permutation | 80 column swaps |
| T5 | Mercury | Uniform poison | 25 columns → all same element |
| T4 | Hematite | Stride pattern | Every 17th column → a+1 |
| T3 | Oleander | Block poison | 50 consecutive → random vectors |
| T2 | Gas Chambers | Selective kill | High-diversity columns → zero one entry |
| T26 | Sophie | Cross-contamination | A/B partition → mean injection |

## Phase II: 12 Biological Horrors

| # | Name | Origin | Operation |
|---|------|--------|-----------|
| B1 | Prion | Misfolding | Chain reaction: j → (j×843)%N |
| B2 | Cordyceps | Mind control | High-connectivity columns → uniform vector |
| B3 | Emerald Wasp | Parasitic injection | Apply T_fake to 40 columns |
| B4 | Tetrodotoxin | Paralysis | 10 blocks → low-rank subspace |
| B5 | Ichneumonid | Low-weight targeting | Lightest 60 columns → flip |
| B6 | Cone Snail | Reversal | 30 pairs → row-reversed copy |
| B7 | Gympie | Local reaction | Adjacent recurrence |
| B8 | Ophiocordyceps | Subtle control | 5% of valid points → single flip |
| B9 | Pit Viper | Heat-seeking | SHA-derived per-column perturbation |
| B10 | Honeybee | Broad sting | 10% of all columns → random flip |
| B11 | Toxoplasma | Disguise | Invalid columns → valid-looking points |
| B12 | Pitcher Plant | Gradient lure | 15 cascading sequences |

## Phase III: 20 Anti-Quantum Defenses

| # | Name | Operation |
|---|------|-----------|
| Q1 | Grover Overshoot | Apply fake T to partial rows |
| Q2 | Shor False Period | Periodic stride perturbation |
| Q3 | Blind Oracle | Rare bit-flip inversion |
| Q4 | Uncompute Trap | Triple linear combination |
| Q5 | Phase Flip | Sparse conditional flip |
| Q6 | Leakage | 20 columns → fixed element |
| Q7 | Crosstalk | Adjacent pair cross-contamination |
| Q8 | Zeno Freeze | 5-column constant runs |
| Q9 | Thermal Ghost | 5% random perturbation |
| Q10 | Monogamy | 25 cross-column additions |
| Q11 | Poisoned Cat | SHA(column) → replace column |
| Q12 | Amplitude Sink | 5 fake attractors × 8 columns each |
| Q13 | Eigenvalue Collision | Triple-column duplication |
| Q14 | Hadamard Haloclina | a ↔ a+1 swap |
| Q15 | Decoherence Abyss | 8-column groups → shared perturbation |
| Q16 | Hydra | Kill one → spawn two |
| Q17 | Quantum Tar Pit | Adjacent averaging |
| Q18 | Siren | True T applied with noise |
| Q19 | Heisenberg Razor | Cross-column transfer |
| Q20 | Heat Death | Full column randomization |

## Phase IV: 5 Structural Evils

| # | Name | v9.3 Status | Operation |
|---|------|-------------|-----------|
| E1 | **Gaslight** | **PRF-deterministic** (FIX-3) | SHA(seed+column+index) → unique injection. Zero collisions. |
| E2 | Gröbner Tar | Active | 35 columns → random values |
| E3 | Sisyphus Fractal | Active | Repeating pattern blocks |
| E4 | Mark of Cain | Active | 7 forensic fingerprint blocks |
| E5 | Basilisk | Active | 50 columns → binary {a, a+1} |

## Phase V: 4 Existential Terrors

| # | Name | Operation |
|---|------|-----------|
| X1 | Turing Horizon | 10 convergence sequences (20 columns each) |
| X2 | Kolmogorov Void | 82 incompressible ghost-pattern columns |
| X3 | Gödel Lock | 30 circular hash-dependency pairs |
| X4 | Ψ·Σ Armageddon | 35 dual-allegiance columns (T + T_fake) |

## Structural Mechanisms (v9.1+)

| # | Name | Version | Status |
|---|------|---------|--------|
| S1 | **Anti-Collision Sweep** | v9.1 | Iterative PRF until 0 duplicate columns |
| S2 | **Monolithic T** | v9.1 | No block decomposition (auditor consensus) |
| S3 | **Conic Regulus** | v9.1 | Hall spread via conic-based regulus (5 lines) |

## Security Models (v9.3)

| Model | Private Key | Security Basis | Status |
|-------|------------|----------------|--------|
| **A** | T = GF(16) multiplication | 273/273 invariance + corruption | Clean attack polynomial |
| **B** | 273 line indices (no T) | Hidden spread + noise. Gap=0.05 | **INDISTINGUISHABLE** |

## Empirical Results (v9.3)

```
Total defenses deployed:     70 traps + 3 structural + 2 models
Corruption:                  37.8% (3094/8190)
Entropy:                     1.994 bits (max 2.0)
Gaslight collisions:         0/100
T-invariance (Model A):      273/273
Model B distinguisher gap:   0.05 (indistinguishable)
Noisy decrypt success:       20/20 at 50% noise
```

---

*70 defenses. 5 phases. 3 rounds of audit. 0 mercy.*  
*The truth is more important than the dream.*
