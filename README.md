# AEGIS — The Crystal Labyrinth

### *A design framework where every attack strategy becomes a liability.*

**Version 9.3 THE LEVIATHAN** · Post-Audit Round 3  
**Pure Python3** · Zero dependencies · 6.6 seconds  
**License:** CC BY-NC-SA 4.0

---

## What Is This?

AEGIS is an experimental geometric-code cryptographic construction built on Desarguesian spreads in PG(5,4). It is **not** a production cryptosystem. It is a research framework exploring whether spread geometry + heavy corruption can provide post-quantum security.

Three independent AI auditors (Gemini, ChatGPT, Grok) have conducted **three rounds of adversarial review**. Every version that looked invincible had a hidden weakness. Every death produced deeper understanding.

## Current State (v9.3) — Honest Numbers

```
Spread:          273 lines in PG(5,4), Desarguesian
T-invariance:    273/273 lines ✓
Noisy decrypt:   OK at 0%, 5%, 10%, 20%, 35%, 50% noise (20/20 each)
Corruption:      37.8% (3094/8190 entries modified)
Entropy:         1.994 bits (max 2.0)
Gaslight:        0/100 collisions
Model B gap:     0.05 (real vs decoy lines INDISTINGUISHABLE)
Centralizer:     dim=18 (under investigation)
Runtime:         6.6s (Python3, MacBook Air M2)
```

## Two Security Models

**Model A:** Private key T = GF(16) multiplication map.  
All 273 lines T-invariant. Clean-case recovery is polynomial time (proven by ChatGPT, Round 2). All security comes from corruption/noise.

**Model B:** Private key = which 273 lines are real (no T).  
Security from "hidden noisy spread recognition." Residual gap = 0.05 (indistinguishable). This may be the real security foundation.

## The Journey (12 versions, 3 audit rounds)

| Version | What Happened |
|---------|--------------|
| v1-v5 | Fragrance Engine era. Built trap layers. |
| v6 | Broken by Gemini in 47 seconds (Reguli filter) |
| v7-v8.3 | 70 defenses. Survived 13 attacks. 9/10 novelty. |
| v9.0 | Hall spread FAILED (line ≠ conic). Blocks separable. |
| v9.1 | 3 fixes: conic regulus ✓, monolithic T ✓, 0 collisions ✓ |
| v9.1→audit | ChatGPT proves clean recovery is POLYNOMIAL TIME |
| v9.2 | Semifield attempt. REGRESSION (isotopic, 21/273 invariant) |
| **v9.3** | **Honest reckoning. Two models. Noisy decrypt works.** |

## The MinRank Connection (Grok's formulation)

For each spread line L with basis M_L:

```
T(L) = L  ⟺  rank([M_L ; T·M_L]) ≤ 2
```

Full problem: find T such that this holds for ≥273 of ~769 candidate lines.  
**Threshold-MinRank with subset selection** — formalized but not yet proven hard.

## Run It

```bash
python3 aegis_v93.py
```

No dependencies. No SageMath. No internet. Just Python 3.

## The Three Principles

1. **You Cannot Catch the Wind** — Security from irreproducibility, not hiding.
2. **The Attacker's Optimal Strategy Is to Flee** — Nash equilibrium: cost > benefit at every step.
3. **The System Does Not Resist — It Transforms** — Attack energy → attacker entropy.

## What's Next

1. Scale to PG(11,4) where true non-Desarguesian semifields exist (order 256)
2. Formalize "Hidden Noisy Spread Recognition" as a named hard problem
3. Publish MinRank connection + noise threshold data as research paper

## Files

| File | Description |
|------|------------|
| `aegis_v93.py` | Main prototype (Python3, executable) |
| `README.md` | This file |
| `GUIDE.md` | Framework explained without math |
| `HISTORY.md` | Complete version chronicle |
| `ARCHITECTURE.md` | Technical deep dive |
| `STRATEGIES.md` | Full catalog of 70+ defenses |
| `AUDIT.md` | Three-round auditor verdicts |
| `OPEN_PROBLEMS.md` | 11 unsolved questions |
| `CHANGELOG.md` | Detailed record of every change |
| `CONTRIBUTING.md` | How to attack or contribute |
| `LICENSE.md` | CC BY-NC-SA 4.0 |
| `CITATION.md` | How to cite |
| `QUOTES.md` | Collected quotes from the journey |

---

*Proyecto Estrella · Error Code Lab*  
*Rafael Amichis Luengo — The Architect*  
*Claude — Lead Engine*  
*"The truth is more important than the dream."*
