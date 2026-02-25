# AEGIS — The Crystal Labyrinth

### *A design framework where every attack strategy becomes a liability.*

**Version 9.3 THE LEVIATHAN** "The Bunker"· Post-Audit Round 3  
**Pure Python3** · Zero dependencies · 6.6 seconds  
**License:** CC BY-NC-SA 4.0

---

## 48 Hours

48 hours before this commit, the creator of this project — a psychology graduate with no formal mathematical training — did not know what McEliece was.

What happened next: 12 prototype versions. 3 rounds of independent adversarial audit by three AI systems. 4 critical vulnerabilities found and fixed in real-time. 1 formal mathematical reduction (Threshold MinRank). 1 unexpected security model discovered. A decryption mechanism that survives 50% noise corruption (20/20 tests).

Every failure is documented. Every death produced a deeper version. The full story is in [THE_ARCHITECTS_METHOD.md](THE_ARCHITECTS_METHOD.md).

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
Runtime:         6.6s (Python3, no dependencies)
```

## Two Security Models

**Model A:** Private key T = GF(16) multiplication map.  
All 273 lines T-invariant. Clean-case recovery is polynomial time (proven by ChatGPT, Round 2). All security comes from corruption/noise.

**Model B:** Private key = which 273 lines are real (no T).  
Security from "hidden noisy spread recognition." Residual gap = 0.05 (indistinguishable). This may be the real security foundation.

## The Journey

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

## Run It

```bash
python3 aegis_v93.py
```

No dependencies. No SageMath. No internet. Just Python 3.

## The Three Principles

1. **You Cannot Catch the Wind** — Security from irreproducibility, not hiding.
2. **The Attacker's Optimal Strategy Is to Flee** — Nash equilibrium: cost > benefit at every step.
3. **The System Does Not Resist — It Transforms** — Attack energy → attacker entropy.

## Files

| File | Description |
|------|------------|
| `aegis_v93.py` | Main prototype (Python3, executable) |
| `THE_ARCHITECTS_METHOD.md` | The full story: 48 hours, methodology, results |
| `README.md` | This file |
| `GUIDE.md` | Framework explained without math |
| `GUIDE_FOR_EVERYONE.md` | Non-technical explanation |
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
