# AEGIS — Open Problems

### Unsolved Questions for Researchers

---

These are genuine open problems that emerged from building and breaking AEGIS. We don't know the answers. Some may be easy. Some may be impossible. All are interesting.

---

## Critical (Must Solve for Cryptosystem Viability)

### OP-1: Reguli-Resistant Decoy Generation

**Problem:** Can non-Desarguesian spreads (Hall, Kantor, semifield) serve as decoys that resist the Reguli distinguisher?

**Context:** The Reguli attack filters Desarguesian spread lines from random decoys in polynomial time by testing whether triplets of lines form geometrically closed subsets. Non-Desarguesian spreads have their own Reguli structure, which might resist this filter.

**Specific question:** Given a Desarguesian spread S and a Hall spread S' in PG(5,4), can an attacker distinguish lines of S from lines of S' in polynomial time?

**Why it matters:** If the answer is "no," the Hall of Mirrors layer becomes cryptographically sound. If "yes," decoy-based obfuscation fundamentally doesn't work.

### OP-2: Layer 2 Coupling Hardness at 2^256

**Problem:** Design a key-splitting scheme where recovering the full key from the split components requires ≥ 2^256 classical operations.

**Current state:** T split into (T₁, T₂) with bilinear coupling B(T₁, T₂). At PG(11,4), brute force = 2^72. Grover halves to 2^36.

**Proposed direction:** Four 3×3 blocks with independent bilinear/trilinear forms plus hash-derived coupling mask. Needs formal analysis.

**Specific question:** What is the minimum number of independent coupling forms needed to achieve 2^256 classical security in PG(11,4)?

### OP-3: Corrupted Spread MinRank Reduction

**Problem:** Define the Corrupted Spread MinRank Problem formally and prove it is at least as hard as standard MinRank (NP-hard).

**Definition sketch:** Given a matrix M over GF(q) that is the sum of a spread-generating matrix and a structured corruption matrix, recover the low-rank spread generator.

**Why it matters:** Without a reduction to a known hard problem, AEGIS is structural obfuscation. With this reduction, it becomes publishable.

---

## Structural (Important for Soundness)

### OP-4: Trap Composition Analysis

**Problem:** Do some of the 70 traps cancel each other? Specifically, does Toxoplasma (disguise broken as healthy) undo Suction Mud (zero columns)?

**Context:** The v8.3 result showing zero=0 in trap scan suggests some cleanup is already happening. Formal analysis of trap interaction could prune to an optimal core set of 15-20 maximally impactful traps.

### OP-5: Noise Distribution Theory over GF(4)

**Problem:** What is the optimal noise distribution for GF(4) matrices that maximizes attacker confusion while preserving owner decodability?

**Specific question:** Given a code of rate R over GF(4), what noise density ε maximizes the gap between owner's decoding probability and attacker's decoding probability?

### OP-6: Fold Indistinguishability

**Problem:** Can the attacker detect Layers 6-7 (the fold/mirage) before committing computational resources to them?

**Context:** If the attacker can identify the fold from outside, they avoid the loop entirely. If they can't, each entry into the fold costs O(n³) with no information gain.

---

## Scaling (Required for Production)

### OP-7: PG(11,4) Implementation

**Problem:** Scale AEGIS from PG(5,4) [1,365 points] to PG(11,4) [5,592,405 points].

**Subproblems:**
- Key generation time at 5.5M points
- Decoy generation: millions of partial spreads needed
- Decryption performance with 1.1M spread lines
- Memory footprint optimization (estimated 33MB for H)
- Constant-time implementation for side-channel resistance

### OP-8: Decoy-at-Scale Generation

**Problem:** Generate algebraically plausible decoy lines at PG(11,4) scale using keyed PRNG — fast enough for practical key generation.

**Constraint:** Decoys must resist Reguli filtering (see OP-1), overlap detection, and statistical profiling.

---

## Theoretical (For the Ambitious)

### OP-9: STSP Hardness in the Obfuscated Setting

**Problem:** Is STSP (Spread Trapdoor Selection Problem) hard when the spread equations are corrupted by structured noise?

**Context:** STSP without corruption is polynomial (the linear algebra attack that broke v6). The question is whether Layers 0-7 raise the complexity from polynomial to exponential.

**This is the fundamental open question of AEGIS.**

### OP-10: Nash Equilibrium Formalization

**Problem:** Formally prove that the attacker's optimal strategy against AEGIS v8.3+ is not to attack.

**Requires:** Defining cost functions for each attack path, computing expected payoffs under trap uncertainty, and showing all paths have negative expected value.

### OP-11: Decoder-Induced Trapdoor Geometry

**Problem:** Formalize the concept of "decoder-induced trapdoor geometry" — where the trapdoor is not in the structure but in how the decoder interacts with the structure.

**Context:** All three auditors identified this as a genuinely unexplored concept in the literature. It emerged from AEGIS Round 14 and may have applications beyond this specific system.

---

## How to Contribute

If you work on any of these problems, we would love to hear from you — whether you solve them, prove them impossible, or find something unexpected along the way.

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to share results.

---

<p align="center">
  <em>"Without attacks it doesn't survive. Can a lion not kill and live?"</em>
</p>
