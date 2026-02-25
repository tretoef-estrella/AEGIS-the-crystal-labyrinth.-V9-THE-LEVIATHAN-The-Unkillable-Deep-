# AEGIS — Technical Architecture

### Finite Fields, Projective Geometry, and Weaponized Matrices

---

## Prerequisites

Familiarity with: finite fields (GF(q)), linear algebra, basic projective geometry. For a non-technical overview, see [GUIDE.md](GUIDE.md).

---

## Algebraic Foundation

### GF(4) and GF(16)

```
GF(4) = GF(2)[a] / (a² + a + 1)
Elements: {0, 1, a, a+1}

GF(16) = GF(4)[z] / (z² + z + a)
Elements: pairs (c₀, c₁) representing c₀ + c₁·z
Minimal polynomial: z² + z + a = 0  →  z² = z + a

Multiplication in GF(16):
  (x₀+x₁z)(y₀+y₁z) = (x₀y₀ + x₁y₁a) + (x₀y₁ + x₁y₀ + x₁y₁)z
```

### Projective Space PG(2n-1, q) via Field Reduction

Each point in PG(n-1, q²) corresponds to a **line** in PG(2n-1, q) with q+1 points. These lines form a **Desarguesian spread**: a perfect partition of all projective points into disjoint lines.

```
Space         Points                 Lines    Pts/Line
PG(5,4)       1,365                  273      5
PG(7,4)       21,845                 4,369    5
PG(11,4)      5,592,405              1,118,481  5
```

### The T Operator

T represents multiplication by z ∈ GF(16) on GF(16)³ viewed as GF(4)⁶.

```
Block-diagonal structure (3 blocks of 2×2):

T = ⎡ 0  a │ 0  0 │ 0  0 ⎤
    ⎢ 1  1 │ 0  0 │ 0  0 ⎥
    ⎢ 0  0 │ 0  a │ 0  0 ⎥
    ⎢ 0  0 │ 1  1 │ 0  0 ⎥
    ⎢ 0  0 │ 0  0 │ 0  a ⎥
    ⎣ 0  0 │ 0  0 │ 1  1 ⎦

Property: T² = T + aI  (minimal polynomial of z over GF(4))
```

A spread line L is **T-closed** if T maps span(L) to itself. For the Desarguesian spread constructed from GF(16), all 273 lines are T-closed for the correct T. For random T: 0/500 lines are closed.

---

## The Key Discovery: Visible but Irreproducible

### What Works

The **frequency gap** is real and dramatic:

| Configuration | T-closed lines |
|--------------|---------------|
| True spread + True T | 273/273 (100%) |
| False spread + True T | 0–2/273 (~0.4%) |
| Any spread + Random T | 0/500 (0%) |

This is the **fingerprint**: a measurable property that perfectly separates the owner's key from everything else.

### What Broke

The spread-stabilizing condition produces a LINEAR system:

```
For each line L with basis {v, w}:
  For each z orthogonal to span(v, w):
    For each point u on L:
      z · (T · u) = 0    ← LINEAR in T's entries
```

At PG(5,4): 5460 equations, 36 unknowns, rank 34, kernel dimension **2**.

The kernel IS GF(16). T recovered in 47 seconds by linear algebra.

**This attack scales to any dimension.** At PG(11,4) the system would be even more overdetermined. The kernel remains small.

### Why It Matters

The frequency mechanism is elegant and real — but it cannot stand alone. The spread gives the attacker too much information. AEGIS v7+ wraps this mechanism in layers that prevent the attacker from using the spread equations cleanly.

---

## v7+ Architecture: 8 Layers

### Layer 0 — Quicksand

Public matrix H' = H + E where noise E has structured correlations.

- Noise density: 15%, keyed by private seed
- Noise-to-signal ratio grows as √n with exploration depth
- Legitimate decoder subtracts E using key

### Layer 1 — Hall of Mirrors

273 real spread lines hidden among ~5000 decoy lines.

- Decoys include 20 partial spreads of ~200 mutually disjoint lines
- Overlap gap: 0.016 (indistinguishable at 1000 samples)
- Selection complexity: C(5000, 273)

**Critical open problem:** Desarguesian spread lines can be distinguished from random decoys via the Reguli test. Fix requires non-Desarguesian decoys (Hall, Kantor spreads).

### Layer 2 — Double Lock

Private key split into (T₁, T₂, C_coupling).

- T₁ operates on coordinates 0–2
- T₂ operates on coordinates 3–5
- Coupled via secret bilinear map B(T₁, T₂)
- Decryption requires B — not decomposable

**Critical open problem:** At PG(11,4), coupling brute force = 2^72. Grover halves this to 2^36. Need 2^256 minimum.

### Layer 3 — Crystal Key

Nonce-based ciphertext binding.

- Each encryption uses one-time permutation
- Wrong decryption produces plausible garbage, not detectable failure
- Attacker has one shot

### Layer 4 — Oil on Glass

Error propagation in attacker's equation system.

- Noise from Layer 0 amplifies when equations are combined
- With 5460 equations (needed for v6 attack), corruption exceeds signal

### Layer 5 — Silver Bridge (Honeypot)

False solution T* planted as attractor.

- Satisfies ~80% of spread equations
- Carries unique fingerprint for traitor tracing
- The "Score Principle" — attacker finds a valid-looking wrong answer

### Layers 6+7 — The Fold and The Mirage

Apparent shortcuts redirect to Layer 2 with degraded state.

- Attack space folds on itself
- Creates loop 2→7→6→2
- Each cycle costs O(n³), accumulates O(√n) noise
- Only escape: absorb full brute force cost of Layer 1

---

## The STSP Hard Problem

```
SPREAD TRAPDOOR SELECTION PROBLEM (STSP):

Given:  H_pub (2n × N matrix over GF(q))
        S = Desarguesian spread (publicly computable)
Find:   T ∈ GL(2n, q) such that every line of S is T-invariant

Candidate count: |GL(2n,q)| / |GL(n,q²)|

PG(5,4):    ~2^35.6   (toy)
PG(7,4):    ~2^72     (getting there)
PG(11,4):   ~2^143.6  (post-quantum range)
PG(31,4):   ~2^2000+  (overkill)

STATUS: Solvable in polynomial time without obfuscation.
        With obfuscation (Layers 0-7): open question.
```

### Candidate Formal Reduction: Corrupted Spread MinRank

Recovering the low-rank multiplication table T inside a corrupted geometric matrix. Strictly harder than plain Code Equivalence. This is the most promising path to a formal security proof.

---

## Empirical Results Summary

### v6 (Frequency Pure — BROKEN)
```
Runtime:    0.8s
Frequency:  273/273 vs 0/500 — WORKS
Attack:     47s — T RECOVERED
```

### v8.3 (The Beast — 70 Defenses)
```
Runtime:     25.2s
Corruption:  67.2%
Entropy:     1.524 bits (max 2.0)
Decryption:  SUCCESSFUL
Attacks:     13/13 contained
```

---

## Running the Prototype

The v8.3 prototype runs in SageMath on any system:

```bash
sage aegis_v8_3_beast_70.sage
```

Expected output: ~25 seconds, ~67% corruption, decryption successful, all 13 attacks contained.

Requirements: SageMath 9.0+ (includes GF(4) and matrix operations).

---

## Scaling Path

| Dimension | Points | Lines | T-space | Layer 2 | Status |
|-----------|--------|-------|---------|---------|--------|
| PG(5,4) | 1,365 | 273 | 2^36 | — | Toy (current) |
| PG(7,4) | 21,845 | 4,369 | 2^72 | 2^72 | Prototype target |
| PG(11,4) | 5.6M | 1.1M | 2^144 | 2^72* | Production target |
| PG(31,4) | ~10^18 | ~10^17 | 2^2000+ | — | Theoretical |

*Layer 2 coupling needs independent hardening to 2^256.

---

<p align="center">
  <em>"The structure is visible. The frequency is inaudible."</em><br>
  <em>— AEGIS v6 epitaph (before the Crystal Labyrinth was built on its grave)</em>
</p>
