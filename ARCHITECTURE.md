# ARCHITECTURE — Technical Deep Dive

## Current State: v9.2 (Semifield Core — experimental, not production)

### Algebraic Foundation

| Parameter | Toy (current) | Target |
|-----------|--------------|--------|
| Base field | GF(4) | GF(4) |
| Ambient | GF(4)^6, PG(5,4) | GF(4)^12, PG(11,4) |
| Points | 1,365 | 5,592,405 |
| Spread lines | 273 | 1,118,481 |
| Private key T | GL(6,4) ≈ 2^71 | GL(12,4) ≈ 2^287 |
| Spread type | Semifield (v9.2) | TBD |

### The Spread Problem (Status: OPEN)

**Desarguesian spread (v9.0–v9.1):**
- T satisfies T²+T+αI=0 (quadratic minimal polynomial)
- Clean-case recovery: POLYNOMIAL TIME (proven by ChatGPT)
- All security comes from corruption (noise), not structure

**Semifield spread (v9.2):**
- T²+T+αI=0 = FALSE (good)
- But: current construction is isotopic to GF(16) (bad)
- Only 21/273 lines T-invariant (debated: feature vs bug)
- Tensor decomposition attack still viable (~2^40-80)

**Needed for v9.3:**
- Proper semifield with identity from classification tables
- T must stabilize ALL lines, or T dropped entirely
- Resistance to isotopy reduction and tensor decomposition

### The MinRank Connection (Grok's formulation)

For each spread line L with basis M_L:

```
T(L) = L  ⟺  rank([M_L ; T·M_L]) ≤ 2
```

Full problem: find T satisfying this for ≥273 of ~500+ candidate lines.
This is "Threshold-MinRank with subset selection" — harder than classical MinRank (combinatorial) but easier due to geometric structure.

### Corruption Engine (70 traps, 5 phases)

- Phase 0: Entropy Collapse (statistical flooding)
- Phase I: 26 Ancient Traps (combinatorial perturbations)
- Phase II: 12 Biological Horrors (structural mimicry)
- Phase III: 20 Anti-Quantum (decoherence simulation)
- Phase IV: 5 Structural Evils (Gaslight PRF, Gröbner tar, Mark of Cain)
- Phase V: 4 Existential Terrors (Turing, Kolmogorov, Gödel, Ψ·Σ)
- Final: Anti-collision sweep (iterative PRF, 0 duplicate columns)

### Decryption (Owner Path)

**Current (v9.1-v9.2):** Uses clean Hc + known spread indices.
Two-error syndrome decoded by exhaustive pair search on known lines.
341 candidates, original always found.

**Problem:** Does not demonstrate noisy decryption from public H.
This is a BLOCKING issue (see OPEN_PROBLEMS OP-2).

### Security Assessment (Post Round 3)

| Attack | v9.1 (Desarguesian) | v9.2 (Semifield) |
|--------|---------------------|-------------------|
| Extension field recovery | Polynomial (clean) | Blocked (no minpoly) but isotopy viable |
| Tensor decomposition | ~2^40-70 | ~2^40-80 |
| Subspace clustering | ~2^60-90 | ~2^60-90 |
| Statistical consistency | N/A | ~2^25-35 (new attack) |
| Graph matching | N/A | < 2^20 at toy (new attack) |

**Bottom line:** No attack reaches 2^256 at any scale. Formal hardness reduction does not yet exist.

*Architecture documented honestly. The lighthouse needs deeper foundations.*
