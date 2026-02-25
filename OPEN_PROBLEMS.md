# OPEN PROBLEMS — AEGIS Post-Audit Round 3

## Critical (must solve for v9.3)

### OP-1: Proper Semifield Construction
Build a genuine non-Desarguesian semifield spread at order 16 (or 64) using explicit multiplication tables from the classification literature (Knuth, Dickson, Albert). The current isotopic scramble is recoverable. Must have identity element and all spread lines T-invariant.
**Status:** BLOCKING. All three auditors agree current construction is wrong.

### OP-2: Noisy Decryption
Current decryption uses clean Hc, not public corrupted H. Must demonstrate error-correction from noisy matrix using only the private spread (list-decoding or iterative correction). Without this, the system is not a functional cryptosystem.
**Status:** BLOCKING. Grok: "not even functional as code-based cryptosystem."

### OP-3: Centralizer Dimension Test
Solve XT = TX over GF(4). If dim(Centralizer(T)) > 1, hidden-field structure survives regardless of semifield choice. ChatGPT: "This single test tells more than all entropy metrics combined."
**Status:** BLOCKING. Not yet computed.

### OP-4: T-Invariance Completeness
In v9.2, only 21/273 lines are T-invariant (down from 273/273 in Desarguesian). Either restore full invariance with a proper semifield, or drop T entirely and prove security from hidden-spread recognition alone.
**Status:** BLOCKING. Grok/ChatGPT say regression; Gemini says consequence of non-associativity.

## Important (needed for publication)

### OP-5: Spread-MinRank Formalization
Formalize "Corrupted Threshold Spread-MinRank" as a named hard problem. Prove or conjecture asymptotic hardness. Grok provided the rank([M_L; T·M_L]) ≤ 2 formulation.
**Status:** Formulation exists. Proof missing.

### OP-6: Tensor Decomposition Resistance
ChatGPT identifies bilinear tensor decomposition as the dominant attack at all scales. Prove that the semifield multiplication tensor is not efficiently decomposable (or that corruption defeats it).
**Status:** Not addressed.

### OP-7: Scale to PG(7,4) or PG(9,4)
Toy scale PG(5,4) is too small for meaningful security estimates. Need prototype at PG(7,4) minimum with concrete keygen/encrypt/decrypt + attack cost estimates.
**Status:** Not started.

### OP-8: Hidden Spread Recognition Hardness
If T is dropped, security reduces to: "given noisy coordinates + candidate lines, identify the 273-line partition." Is this problem hard? Grok estimates < 2^35 at toy via statistical consistency. Needs formal analysis at scale.
**Status:** New problem identified in Round 3.

## Research Questions

### OP-9: Semifield Classification at Order 16
The complete classification of semifields of order 16 is known. Which ones produce spreads where left-multiplication stabilizes all lines? Which resist isotopy reduction?

### OP-10: Corruption as Hardness Amplifier
ChatGPT: "Corruption acts as heuristic obfuscation, not hardness reduction." Can we formalize corruption's role, analogous to noise in LPN/LWE?

### OP-11: Publication Venue
All three auditors recommend: publish as cryptanalysis/finite-geometry paper first, not as cryptosystem. Target venues: Designs Codes and Cryptography, PQCrypto workshop, IACR ePrint.

*11 open problems. 4 blocking. The truth is more important than the dream.*
