# Contributing to AEGIS

---

## Welcome

AEGIS is an open research project. Whether you're a cryptographer, mathematician, security engineer, or curious explorer — contributions are welcome.

---

## What We Need Most

### 1. Attacks

The best contribution is breaking something. If you find a vulnerability, a shortcut, or a way to bypass any defense layer — that's not a failure, it's the most valuable input possible.

> *"Without attacks it doesn't survive. Can a lion not kill and live?"*

### 2. Open Problem Solutions

See [OPEN_PROBLEMS.md](OPEN_PROBLEMS.md) for the full list. The three most impactful:

- **OP-1:** Can non-Desarguesian spread decoys resist the Reguli filter?
- **OP-3:** Does Corrupted Spread MinRank reduce to a known NP-hard problem?
- **OP-9:** Is STSP hard in the obfuscated setting?

### 3. Implementations

- SageMath prototype at production scale (PG(11,4))
- Rust/C++ implementation with constant-time operations
- Formal verification of trap composition (OP-4)

### 4. Connections to Existing Literature

If you recognize a mechanism, technique, or problem from existing work — please share the reference. We may have reinvented something, and knowing the prior art strengthens the project.

---

## How to Contribute

### For Attacks and Analysis

Open an issue titled `[ATTACK] Brief description` with:
- Which layer/mechanism you targeted
- Your approach
- Results (successful break, partial, or unsuccessful with reasoning)
- Suggested fix if applicable

### For Research Results

Open an issue titled `[RESEARCH] OP-N: Brief description` with:
- Which open problem you addressed
- Your result (solution, impossibility proof, partial progress, or reformulation)
- Supporting materials (proofs, code, references)

### For Implementations

Open a pull request with:
- Clear description of what's implemented
- Test results
- Performance benchmarks

### For Literature Connections

Open an issue titled `[REFERENCE] Brief description` with the relevant citation and how it connects to AEGIS.

---

## Standards

- **Honesty over diplomacy.** If something is broken, say so directly. We handle that well.
- **Show your work.** Claims need evidence — code, proofs, or clear reasoning.
- **Respect the philosophy.** The 12 Laws and the design principles aren't decorative. They encode real design decisions.
- **English for all technical content.** The project operates in English for maximum accessibility.

---

## What We Won't Accept

- Security claims without evidence
- Changes that hide known vulnerabilities
- Commercial forks (see [LICENSE.md](LICENSE.md))

---

## Recognition

All contributors who find vulnerabilities, solve open problems, or make significant contributions will be credited in the project documentation.

---

<p align="center">
  <em>"The truth is more important than the dream."</em><br>
  Proyecto Estrella · Error Code Lab
</p>
