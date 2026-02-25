# CHANGELOG — The Complete Record

*Every change. Every death. Every resurrection. This is history.*

---

## v9.3 — THE LEVIATHAN (Proper Reckoning) — 25 Feb 2026

### Fixes (4 blocking issues from Round 3)
- **FIX-6:** Honest spread construction. Acknowledged that all order-16 semifields are isotopic to GF(16). Desarguesian spread used directly. No false claims.
- **FIX-7:** T-invariance restored to 273/273 (was 21/273 in v9.2).
- **FIX-8:** Noisy decryption demonstrated. Three paths tested:
  - Path 1: Private Hc (standard McEliece) → OK
  - Path 2: Noisy H + spread knowledge → OK  
  - Path 3: Fully noisy spread-guided → OK
  - Noise threshold: 20/20 successful at 0%, 5%, 10%, 20%, 35%, 50%
- **FIX-9:** Centralizer dimension computed. dim(Centralizer(T)) = 18 (under investigation).

### New Features
- **Two Security Models:** Model A (T-based) and Model B (hidden spread only)
- **Model B validation:** Attacker residual gap = 0.05 (indistinguishable)
- **Pure Python3:** No SageMath dependency. Runs anywhere.

### Auditor Status
- All 4 blocking issues addressed
- Pending: Round 4 audit on v9.3 results

---

## v9.2 — Semifield Core — 25 Feb 2026

### Changes
- **FIX-4:** Replaced Desarguesian spread with semifield spread
- **FIX-5:** Noise threshold experiment added
- Result: T²+T+αI=0 = FALSE (quadratic minimal polynomial eliminated)

### Problems Found (Round 3 audit)
- Isotopic scramble ≠ real semifield (all three auditors)
- Only 21/273 lines T-invariant (regression)
- Decryption used clean Hc only (not functional cryptosystem)
- **Grok:** "v9.2 is a regression"
- **ChatGPT:** "Isotopy preserves tensor rank"
- **Gemini:** "Direction correct, implementation wrong"

---

## v9.1 — Post-Audit Round 1 — 25 Feb 2026

### Fixes (3 issues from Round 1)
- **FIX-1:** Conic-based regulus → SUCCESS (5 transversals, 5 lines replaced)
- **FIX-2:** Monolithic T (removed block decomposition) → GL(12,4) ≈ 2^287
- **FIX-3:** PRF Gaslight + anti-collision sweep → 0/100 collisions

### Discoveries (Round 2 audit)
- **Grok:** MinRank formulation: rank([M_L; T·M_L]) ≤ 2
- **ChatGPT:** Clean-case recovery is POLYNOMIAL TIME
- **Gemini:** Semifield spread breaks the polynomial attack

---

## v9.0 — THE LEVIATHAN (First Attempt) — 25 Feb 2026

### New
- Hall spread derivation (FAILED — selected line instead of conic)
- Layer 2 block coupling (VULNERABLE — separable)
- Reguli attack vector (#14)
- 15-vector attack battery (was 13)

### Round 1 Audit Results
- Hall spread: 0 transversals (line ≠ conic in PG(2,16))
- Block T: security additive not multiplicative
- Gaslight: 32/100 collisions (statistical distinguisher)

---

## v8.3 — THE BEAST (70 Defenses) — Feb 2026

- 26 ancient + 12 biological + 20 anti-quantum + 5 structural + 4 existential + 3 meta
- Algorithm of Terror (randomized trap order)
- 13/13 attack vectors survived
- Auditor novelty score: 9/10
- Critical gaps: Reguli filter, Layer 2 weakness, no formal reduction

---

## v6 — First Auditor Contact — Feb 2026

- Broken by Gemini in 47 seconds
- Reguli filter exposed Desarguesian spread structure
- Death cause: spread lines algebraically distinguishable from random

---

## v1-v5 — Fragrance Engine Era — Jan-Feb 2026

- GF(4) spread geometry in PG(5,4) established
- Iterative trap layering developed
- 13-vector attack battery created
- Each version broke, each added defenses
- Foundation of the Crystal Labyrinth laid

---

## The Death Count

| Version | How It Died | What We Learned |
|---------|-------------|-----------------|
| v6 | Reguli filter (47s) | Spreads have detectable algebraic signature |
| v9.0 | Line ≠ conic | PG(2,16) lines ≠ reguli in PG(5,4) |
| v9.1 | Polynomial clean attack | GF(16) multiplication → T²+T+αI=0 → poly time |
| v9.2 | Isotopic scramble | All order-16 semifields isotopic to GF(16) |

**Deaths: 4. Resurrections: 4. Lessons: priceless.**

---

*Proyecto Estrella · Error Code Lab*  
*"The truth is more important than the dream."*
