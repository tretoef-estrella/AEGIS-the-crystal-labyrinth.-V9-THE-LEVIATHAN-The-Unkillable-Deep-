# OPEN PROBLEMS — AEGIS v9.3 Post-Audit Round 3

## SOLVED (previously blocking)

### ~~OP-1: Proper Semifield Construction~~ → RESOLVED (v9.3)
All order-16 semifields are isotopic to GF(16) (Knuth classification). Acknowledged honestly. Using Desarguesian spread directly. Semifield direction requires order 64+ (PG(11,4)).

### ~~OP-2: Noisy Decryption~~ → RESOLVED (v9.3)
Three decryption paths demonstrated: private Hc, noisy H + spread, fully noisy. All OK. Noise threshold tested 0%-50%: 20/20 at every level.

### ~~OP-4: T-Invariance Completeness~~ → RESOLVED (v9.3)
Restored to 273/273 using correct GF(16) field multiplication map. Was 21/273 in v9.2 (isotopic scramble bug).

## BLOCKING (must solve for v10)

### OP-3: Centralizer Dimension
dim(Centralizer(T)) = 18 in v9.3. Expected ~2 for GF(16). Value of 18 means the commutant of T is much larger than expected — possibly the full GF(4)-span of the spread structure. Must investigate: is this exploitable? Does it leak information?
**Status:** BLOCKING. Unexpected result, not yet understood.

### OP-12: Implement Full 18-Vector Attack Battery
15 original attacks + 3 new from Round 3 (tensor decomposition, statistical consistency, graph matching). Must be integrated into v9.3 to validate all claims.
**Status:** BLOCKING. In progress (next chat).

### OP-13: Graph Matching Attack (Grok)
Grok estimates < 2^20 at toy scale via bipartite matching weighted by column consistency. If correct, Model B security collapses at PG(5,4). Must implement and measure.
**Status:** BLOCKING. Could invalidate Model B at toy scale.

## IMPORTANT (needed for publication)

### OP-5: Spread-MinRank Formalization
Formalize "Corrupted Threshold Spread-MinRank" as a named hard problem. Grok's formulation: rank([M_L; T·M_L]) ≤ 2. Proof of hardness missing.
**Status:** Formulation exists. Proof needed.

### OP-6: Tensor Decomposition Resistance
ChatGPT: bilinear tensor decomposition is the dominant attack. Isotopy reduction ~2^30-50. Must prove semifield multiplication tensor is not efficiently decomposable (or that corruption defeats it).
**Status:** Attack proposed, not implemented.

### OP-7: Scale to PG(11,4)
Toy scale PG(5,4) too small for meaningful security. At PG(11,4): true non-Desarguesian semifields exist (order 256), GL(12,4) ≈ 2^287 bits natural security.
**Status:** Not started. Requires significant engineering.

### OP-8: Hidden Spread Recognition Hardness
Model B security = "given noisy coordinates + candidate lines, identify the 273-line partition." Is this formally hard? v9.3 shows gap=0.05 (indistinguishable) but Grok proposes graph matching could solve it.
**Status:** Empirically strong (gap=0.05). Formally unproven.

### OP-10: Corruption as Hardness Amplifier
ChatGPT: "Corruption acts as heuristic obfuscation, not hardness reduction." Can we formalize corruption's role analogous to noise in LPN/LWE? The 20/20 at 50% noise result is strong empirical evidence.
**Status:** Empirical data exists. Formal reduction missing.

### OP-11: Publication Venue
All three auditors recommend: cryptanalysis/finite-geometry paper first. Targets: Designs Codes and Cryptography, PQCrypto workshop, IACR ePrint.
**Status:** Consensus on venue. Paper not written.

## RESEARCH QUESTIONS

### OP-9: Semifield Classification at Order 64/256
At PG(11,4) with GF(256), true non-Desarguesian semifields exist. Which ones resist isotopy reduction? Which produce spreads with full T-invariance?

### OP-14: Model B Key Space Analysis
C(769, 273) is astronomically large but brute force is not the right metric. What is the information-theoretic lower bound for identifying the hidden spread from noisy observations?

### OP-15: Noise Threshold Phase Transition
v9.3 shows 20/20 decrypt success at 50% noise. Is there a sharp phase transition? At what noise level does decryption fail? This determines the operational parameters of the system.

---

*15 open problems. 3 blocking. 3 solved. The truth is more important than the dream.*
