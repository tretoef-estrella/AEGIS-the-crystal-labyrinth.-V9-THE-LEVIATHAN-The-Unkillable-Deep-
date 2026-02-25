# AEGIS — Strategy Catalog

### 70 Defense Mechanisms: Origins, Principles, and Implementations

---

## How to Read This

Each strategy is listed with:
- **Origin** — Where the idea came from (history, biology, physics, mathematics, cinema)
- **Principle** — What it does to the attacker in plain language
- **Effect** — The concrete cryptographic implementation

Strategies are organized by phase and deploy in **randomized order** within each phase (the Algorithm of Terror).

---

## Infrastructure

### Layer -1: Who Invited You?
**Origin:** Access control  
**Principle:** The first 6 columns encode a hash challenge. Without the preimage, you have no business here.  
**Effect:** Entry denial — wrong key can't even begin the attack

### Hall of Mirrors
**Origin:** Fairground optics  
**Principle:** 273 real objects hidden among 5000+ reflections. Which is which?  
**Effect:** Decoy lines indistinguishable from real spread lines; selection complexity C(5000, 273)

### Algorithm of Terror
**Origin:** Guerrilla warfare  
**Principle:** You don't know which trap comes next. The uncertainty is the weapon.  
**Effect:** All trap deployment orders randomized per private key seed

---

## Phase I — The Ancient Gauntlet (26 Traps)

Historical warfare mechanisms mapped to attacks on the attacker's computational state.

| # | Name | Origin | Principle | Effect |
|---|------|--------|-----------|--------|
| 1 | **Chamber of Grace** | AEGIS original | Your success IS your condemnation | Traitor tracing fingerprint in extracted keys |
| 2 | **Siphon Gas Chambers** | Maya cenotes | The air you seek stops your heart | High-information columns poisoned |
| 3 | **Oleander Walls** | Ancient gardens | Air becomes poison with heat | 50-column toxic subspace block |
| 4 | **Hematite Dust** | Egyptian tombs | Lungs turn to stone | Petrification entries across columns |
| 5 | **Mercury Pit** | Qin Shi Huang | Toxic vapor, impassable | All-generator columns (maximum confusion) |
| 6 | **Forgetting Fungus** | Chinese tombs | Erases your mental map | Index scrambles destroy positional information |
| 7 | **Aspergillus Spores** | Egyptian tombs | Dormant, waiting for breath | Cascade trigger triplets |
| 8 | **Radon Emanations** | Egyptian tombs | Invisible curse, die later | Delayed contradiction pairs |
| 9 | **Obsidian False Exit** | Aztec temples | Run toward your reflection | Second attractor T** planted |
| 10 | **Vibration Lintels** | Egyptian tombs | Forcing the door activates it | Spectral decoy columns |
| 11 | **Haloclines** | Maya cenotes | Visual lie hides abyss | Mirror column pairs |
| 12 | **Suction Mud** | Maya cenotes | Visual blackout + trapping | Zero columns (black holes) |
| 13 | **Inverse Tide Wells** | Various | Time is the executioner | Progressive noise gradient |
| 14 | **Suction Siphons** | Maya hydraulics | Tide makes return impossible | Hidden linear dependencies |
| 15 | **Arrow-Tip Floors** | Chinese tombs | Flagstone becomes abyss | Conditional column traps |
| 16 | **Sand Pit** | Egyptian tombs | Time and weight kill | Multiply-corrupted columns fully buried |
| 17 | **Right-Handed Stairs** | Medieval castles | Asymmetric physical advantage | Asymmetric noise entries (top-heavy) |
| 18 | **False Chambers** | Egyptian pyramids | You think you've stolen all | Decoy solution T_fake planted |
| 19 | **False Pivot Doors** | European castles | Mass enters, can't exit | Roach motel clusters |
| 20 | **Nightingale Floors** | Japanese castles | Stealth becomes alarm | Arithmetic progression markers |
| 21 | **False Step Stairs** | Maya pyramids | Attacks proprioception | Column index swaps |
| 22 | **Granite Portcullis** | Egyptian tombs | Wall drops after exit | Late-stage column corruptions |
| 23 | **Dried Gut Cables** | Various | Invisible tightening whip | Prime-indexed column corruptions |
| 24 | **Murder Holes** | Medieval castles | Boiling sand on armor | High-weight columns corrupted |
| 25 | **Bronze Crossbows** | Qin Dynasty | Persistent structural decoy | Column pairs with forced XOR patterns |
| 26 | **Sophie's Choice** | AEGIS original | Kill daughter A or B? | Complementary destruction fork — entangled column sets |

---

## Phase II — Biological Horror (12 Traps)

Nature's most terrifying organisms, translated to mathematical warfare.

| # | Agent | Type | What It Does to the Attacker |
|---|-------|------|------------------------------|
| 1 | **Prion (Kuru/CJD)** | Misfolded protein | Self-propagating corruption — column j infects column (j×843)%N |
| 2 | **Cordyceps** | Fungal parasite | Hijacks highest-connectivity columns as broadcast towers |
| 3 | **Emerald Wasp** | Neurotoxin | Zombie convergence — columns modified to satisfy T_fake closure |
| 4 | **Tetrodotoxin** | Blue-ring octopus | Conscious paralysis — rank-3 clusters stall Gaussian elimination |
| 5 | **Ichneumonid Wasp** | Parasitic larva | Eats low-weight columns first — system looks healthy then collapses |
| 6 | **Cone Snail** | Insulin shock | False symmetry pairs — algebraic hypoglycemia |
| 7 | **Gympie-Gympie** | Plant toxin | Self-consistent triplets (c3=c1+c2) — unfilterable structural pain |
| 8 | **Ophiocordyceps** | Control fungus | Best-looking columns are most poisoned — trusting your instincts kills you |
| 9 | **Pit Viper** | Infrared vision | Watermarked columns for traitor tracing — sees you by your heat |
| 10 | **Japanese Honeybee** | Collective heat | Low-amplitude noise across every column — individually harmless, collectively fatal |
| 11 | **Toxoplasma** | Protozoan | Removes warning signs — broken columns disguised as healthy |
| 12 | **Pitcher Plant** | Carnivorous | One-way computational slope — walls too slippery to climb |

---

## Phase III — Anti-Quantum Decoherence (20 Traps)

Each targets a specific quantum computing primitive.

| # | Name | Creator | Quantum Vulnerability Targeted |
|---|------|---------|-------------------------------|
| Q1 | **Grover Overshoot** | Architect | Columns oscillating between 2 fake attractors |
| Q2 | **Shor False Period** | Architect | Apparent periodicity with irrational dissonance at 3rd cycle |
| Q3 | **Blind Oracle** | Architect | Silent phase inversions (0.1%) |
| Q4 | **Uncompute Trap** | Architect | Residual entanglement garbage (3-column XOR dependencies) |
| Q5 | **Phase Flip** | Architect | Silent sign errors preserving Hamming weight |
| Q6 | **Leakage State** | Architect | Columns outside computational basis |
| Q7 | **Crosstalk** | Architect | Adjacent columns mutually corrupt |
| Q8 | **Zeno Freeze** | Architect | Blocks of identical columns (measurement freezes evolution) |
| Q9 | **Thermal Ghost** | Architect | Noise with thermal profile (indistinguishable from fluctuation) |
| Q10 | **Monogamy** | Architect | Columns forced to entangle with noise environment |
| Q11 | **Poisoned Cat** | Architect | Self-referential columns (reading collapses to garbage) |
| Q12 | **Amplitude Sink** | Architect | 5 fake attractors competing for probability amplitude |
| Q13 | **Eigenvalue Collision** | Architect | Degenerate eigenvalues confuse QPE |
| Q14 | **Hadamard Haloclina** | Architect | Self-conjugate entries that invert under Hadamard |
| Q15 | **Decoherence Abyss** | Architect | Cross-column dependencies requiring impossible coherence |
| Q16 | **The Hydra** | Engine | Cut one dependency = two grow back |
| Q17 | **Quantum Tar Pit** | Engine | 2-dim subspace where amplitude enters but can't leave |
| Q18 | **The Siren** | Engine | Correct solution at wrong phase factor |
| Q19 | **Heisenberg's Razor** | Engine | Complementary pairs: know one, destroy the other |
| Q20 | **Heat Death** | Engine | Maximum entropy blocks: zero extractable information |

**Important note:** The three auditors unanimously agreed that these traps function as **classical obfuscation**, not quantum-specific defenses. A quantum computer compiles H_pub into a Grover oracle — it doesn't "experience" the columns. The real anti-quantum defense is coupling strength (Layer 2), not trap design. We document this honestly while keeping the traps for their classical value.

---

## Phase IV — Structural Evil (5 Traps)

> *"We don't protect the secret. We poison the enemy."*

| # | Name | Principle |
|---|------|-----------|
| 1 | **Semantic Gaslight** | Attacker wins a *perfect lie* — valid-format key producing disinformation |
| 2 | **Gröbner Tar Pit** | Computation reaches 99% then requires exabytes for the last 1% |
| 3 | **Sisyphus Fractal** | Decoding produces a new AEGIS ciphertext — infinite recursion disguised as depth |
| 4 | **Mark of Cain** | Extracted keys carry forensic beacons revealing the attacker's methods |
| 5 | **The Basilisk** | The math attacks the *software* processing it — exponential memory in CAS |

---

## Phase V — Existential Terror (4 Traps)

> *"The cave no longer contains arrows or biological poisons. It contains paradoxes."*

| # | Name | Principle |
|---|------|-----------|
| 1 | **Turing Horizon** | Asymptotic convergence: 99.9%... 99.99%... The limit doesn't exist. The machine never halts. |
| 2 | **Kolmogorov Void** | Maximum complexity disguised as structure. ML detects phantom correlations that vanish on exploitation. |
| 3 | **Gödel Lock** | To verify A you need B. To verify B you need A. No fixed point. The proof exists but can't be proven. |
| 4 | **Ψ·Σ Armageddon** | Columns satisfying BOTH the real and fake key. Any analyzer must hold two mutually exclusive hypotheses. |

---

## Phase 0/Final — Entropy Collapse (Wrapper)

Applied before all traps AND after all traps:

| # | Name | Principle |
|---|------|-----------|
| EC-1 | **Noise Forest** | Max-entropy columns — fire hose of randomness |
| EC-2 | **Cost Function Collapse** | Gradient = 0 everywhere — all optimization algorithms starve |
| EC-3 | **Landauer Heat Trap** | Columns = XOR of 8 sources — verification requires massive bit erasure |

---

## Film-Derived Design Principles

### The Score (2001)
De Niro knows the betrayal is coming. He plants a fake. The thief steals it, leaves happy, discovers it's worthless. Can't report the theft.

**AEGIS translation:** The attacker finds a valid-looking solution among the noise. It satisfies all public checks. It's fake. The attacker acts on disinformation.

### Flawless (2007)
30 years walking corridors. Doesn't crack the safe — flushes diamonds down the plumbing. Infrastructure only the insider knows.

**AEGIS translation:** The owner decodes through the spread geometry — a path that doesn't exist in the public matrix. The attacker looks for a door. The owner uses the pipes.

---

## Statistics

```
Total mechanisms:        70
Historical warfare:      26
Biological horror:       12
Anti-quantum:            20
Structural evil:          5
Existential terror:       4
Entropy wrapper:          3
Infrastructure:           3 (Layer -1, Hall of Mirrors, Algorithm of Terror)
Original corruption:     67.2%
Decryption:              SUCCESSFUL through all phases
```

---

<p align="center">
  <em>"The algorithm of chaos is terrifying: you never know which one comes next."</em>
</p>
