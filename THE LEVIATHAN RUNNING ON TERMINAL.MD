Last login: Wed Feb 25 11:49:25 on ttys003
rafa@MacBook-Air-de-RAFAEL AEGIS-the-crystal-labyrinth.-V9-THE-LEVIATHAN-The-Unkillable-Deep- % python3 aegis_v93_full.py
========================================================================
  AEGIS v9.3 — THE LEVIATHAN (Full Battery · 18 Attacks)
  Fixes: Real semifield | Full invariance | Noisy decrypt | Centralizer
========================================================================
  PG(5,4)... 1365 pts (0.0s)
  Building spread + security models... 273 lines, Model A + Model B (0.1s)
  T-invariance check... Model A: 273/273 lines T-invariant
  Centralizer test... dim=18
  Corruption engine... done (0.0s)
  Corruption: 3178/8190 (38.8%)
  Entropy: 1.995 bits
  Gaslight: 0/100 collisions
  Decoys... 769 total (0.2s)

  --- DECRYPTION TESTS ---
  Path 1 (private Hc): OK (cands=341)
  Path 2 (noisy H + spread): OK (cands=246)
  Path 3 (fully noisy): OK (cands=246)

  --- NOISE THRESHOLD ---
  Noise  0%: 20/20
  Noise  5%: 20/20
  Noise 10%: 20/20
  Noise 20%: 20/20
  Noise 35%: 20/20
  Noise 50%: 20/20

  --- MODEL B: Security without T ---
  Avg residual: real=2.33, decoy=2.39, gap=0.06
  >>> INDISTINGUISHABLE (gap < 0.5)

========================================================================
  18-ATTACK BATTERY
========================================================================

  [1] Algebraic (kernel) attack...
    kernel_hits=0/200 (0.0%), threshold <15%
    >>> DEFENDED

  [2] Oracle (real lines only) attack...
    oracle_correct=106/273, threshold <250
    >>> DEFENDED

  [3] Greedy spread recovery attack...
    greedy_found=88/273 real lines in greedy partition
    >>> DEFENDED

  [4] Overlap distinguisher attack...
    real_avg=3.672, decoy_avg=3.613, gap=0.059
    >>> DEFENDED

  [5] Noise strip attack...
    stripped_correct=11/200 (5.5%)
    >>> DEFENDED

  [6] Trap scan attack...
    duplicates=0, sha_like=0/300
    >>> DEFENDED

  [7] Attractor attack...
    T-cycle_hits=0/200 (0.0%)
    >>> DEFENDED

  [8] Entropy attack...
    avg_column_entropy=1.567 bits (target >1.5)
    >>> DEFENDED

  [9] Gaslight detection attack...
    collisions=0/100 (target <5)
    >>> DEFENDED

  [10] Gödel detection attack...
    self_ref_hits=0 (expected ~1, threshold <5)
    >>> DEFENDED

  [11] Turing convergence attack...
    gradient_patterns=0 (threshold <3)
    >>> DEFENDED

  [12] Psi-Sigma attack...
    mean_dist=4.44, var=1.09 (threshold mean>3.0)
    >>> DEFENDED

  [13] ISD (Information Set Decoding) attack...
    toy ~2^20 (TRIVIAL), real_scale ~2^1274
    >>> VULNERABLE (toy-scale expected)

  [14] Reguli filter attack...
    reguli_found=0/200 (threshold <3)
    >>> DEFENDED

  [15] Monolithic T brute force estimate...
    GL(6,4)=71 bits, GL(12,4)=287 bits
    >>> DEFENDED

  [16] Tensor Decomposition attack (ChatGPT)...
    product_hits=100/100 (100.0%), est. ~2^1 work
    >>> VULNERABLE

  [17] Statistical Consistency attack (Grok)...
    mean_real=2.328, mean_decoy=2.371, Cohen_d=0.046, est. ~2^217
    >>> DEFENDED

  [18] Graph Matching attack (Grok)...
    matched_real=90/273, search_space ~2^716
    >>> DEFENDED

========================================================================
  AEGIS v9.3 THE LEVIATHAN — FULL BATTERY VERDICT (18 attacks)
========================================================================

  CORE INTEGRITY:
  T-invariance:   273/273 (Model A)
  Centralizer:    dim=18
  Decryption:     P1=OK P2=OK P3=OK
  Noise@50%:      20/20
  Model B gap:    0.06
  Corruption:     3178/8190 (38.8%)
  Entropy:        1.995 bits
  Gaslight:       0/100

  ATTACK BATTERY (16/18 DEFENDED):
  ————————————————————————————————————————————————————————————
  [ 1] ✓ Algebraic (kernel)             DEFENDED     | kernel_hits=0/200 (0.0%), threshold <15%
  [ 2] ✓ Oracle (real lines)            DEFENDED     | oracle_correct=106/273, threshold <250
  [ 3] ✓ Greedy spread recovery         DEFENDED     | greedy_found=88/273 real lines in greedy partition
  [ 4] ✓ Overlap distinguisher          DEFENDED     | real_avg=3.672, decoy_avg=3.613, gap=0.059
  [ 5] ✓ Noise strip                    DEFENDED     | stripped_correct=11/200 (5.5%)
  [ 6] ✓ Trap scan                      DEFENDED     | duplicates=0, sha_like=0/300
  [ 7] ✓ Attractor                      DEFENDED     | T-cycle_hits=0/200 (0.0%)
  [ 8] ✓ Entropy                        DEFENDED     | avg_column_entropy=1.567 bits (target >1.5)
  [ 9] ✓ Gaslight detection             DEFENDED     | collisions=0/100 (target <5)
  [10] ✓ Gödel detection                DEFENDED     | self_ref_hits=0 (expected ~1, threshold <5)
  [11] ✓ Turing convergence             DEFENDED     | gradient_patterns=0 (threshold <3)
  [12] ✓ Psi-Sigma                      DEFENDED     | mean_dist=4.44, var=1.09 (threshold mean>3.0)
  [13] ✗ ISD                            VULNERABLE   | toy ~2^20 (TRIVIAL), real_scale ~2^1274
  [14] ✓ Reguli filter                  DEFENDED     | reguli_found=0/200 (threshold <3)
  [15] ✓ Monolithic T brute force       DEFENDED     | GL(6,4)=71 bits, GL(12,4)=287 bits
  [16] ✗ Tensor Decomposition           VULNERABLE   | product_hits=100/100 (100.0%), est. ~2^1 work
  [17] ✓ Statistical Consistency        DEFENDED     | mean_real=2.328, mean_decoy=2.371, Cohen_d=0.046, est. ~2^217
  [18] ✓ Graph Matching                 DEFENDED     | matched_real=90/273, search_space ~2^716

  ————————————————————————————————————————————————————————————
  Original 15:    14/15 defended
  Round 3 (new):  2/3 defended

  HONEST ASSESSMENT:
  Semifield at order 16 → all isotopic to GF(16) (Knuth 1965).
  Clean-case recovery is polynomial (ChatGPT correct).
  Security rests on noise + hidden spread (Grok correct).
  Need order 64+ for non-Desarguesian hardness (Gemini correct).

  THE PATH FORWARD:
  1. Scale to PG(11,4) — order-256 true semifields
  2. Formalize Model B hidden-spread hardness
  3. Publish MinRank connection + noise threshold data

  Runtime: 5.5s
========================================================================
  Proyecto Estrella · Error Code Lab
  Rafa — The Architect / Claude — Lead Engine
  'The truth is more important than the dream.'
========================================================================

rafa@MacBook-Air-de-RAFAEL AEGIS-the-crystal-labyrinth.-V9-THE-LEVIATHAN-The-Unkillable-Deep- % 
