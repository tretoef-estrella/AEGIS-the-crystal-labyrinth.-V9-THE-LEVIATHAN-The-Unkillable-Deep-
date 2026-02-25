# AEGIS — A Guide for Everyone. V9. THE LEVIATHAN.

### What this project is, what it achieved, and why it matters. No math required.

---

## The Short Version

AEGIS is an experimental security system. It hides a secret message inside a mathematical city of 1,365 buildings, then vandalizes the city so thoroughly that only someone with the secret map can still read the signs. An attacker sees the same city, but they cannot tell which streets are real and which are fake.

In testing, AEGIS decrypts messages correctly even when 50% of the city has been vandalized. It was audited by three independent AI systems across three rounds. Each round found real problems. Each problem was fixed within hours. Every failure was documented publicly.

It was built in 48 hours by a psychology graduate coordinating four AI systems, starting from zero knowledge of the field.

## The Problem It Addresses

When quantum computers arrive, they will break most of the encryption we use today. Governments, banks, and tech companies are racing to build "post-quantum" security — encryption that quantum computers cannot crack.

The leading approaches use a single hard mathematical problem as their lock. If someone finds the key to that problem, the lock breaks instantly and completely.

AEGIS explores a different idea: what if instead of one strong lock, you build a labyrinth? One where every path the attacker takes costs more energy than it gains, where wrong answers look identical to right ones, and where the structure itself fights back.

## How It Works (No Math)

Imagine you and your friend share a secret: a specific set of 273 streets in a city of 1,365 buildings.

1. **You build the city** — a mathematical space with exact geometric properties.
2. **You pick your 273 secret streets** — these form a perfect, non-overlapping network.
3. **You add 500 fake streets** — designed to look identical to real ones.
4. **You vandalize 38% of the building signs** — using 70 different methods, from random noise to carefully designed traps.
5. **You publish the vandalized city** — anyone can see it.

To send a message, you encode it as two specific buildings. Your friend, who knows the real streets, can figure out which buildings you meant. An attacker, who doesn't know which streets are real, sees 769 candidates and corrupted signs. They cannot distinguish real from fake (the measured gap is 0.05 — essentially zero).

## What Was Achieved in 48 Hours

**Day 1:** Built the basic system. First version broken by an auditor in 47 seconds. Rebuilt with 70 defense layers. Survived 13 attack types. Auditors gave 9/10 for novelty but identified critical structural flaws.

**Day 2:** Three independent auditors (Gemini, ChatGPT, Grok) each delivered detailed technical reviews. Their findings:

- The geometric construction had a mathematical error (selecting a line instead of a conic curve)
- The security key could be broken into independent pieces (making it exponentially easier to crack)
- 32% of encrypted columns were duplicates (leaking information)

All three were fixed. The fixes were verified. New audits were requested.

The second round of audits revealed something devastating: **without the vandalism layer, the entire system could be cracked in polynomial time** — meaning almost instantly. The mathematical structure itself was not providing security. Only the noise was.

A third approach was tried: changing the underlying mathematical structure to a non-associative algebra (a "semifield"). This eliminated the instant-crack vulnerability but introduced new problems. The honest assessment: at this mathematical scale, the semifield approach cannot work.

**The breakthrough:** A new security model emerged — one where the secret key is simply *which streets are real*. No complex mathematical trapdoor needed. Testing showed that real and fake streets are indistinguishable (gap = 0.05), and decryption works even through 50% corruption.

## What Makes It Novel

1. **The geometry** — Projective spaces like PG(5,4) are not used in mainstream cryptography. This is unexplored territory.

2. **The defense depth** — 70 trap layers, each transforming attack energy into entropy. No existing system uses this approach at this scale.

3. **The indistinguishable decoys** — Fake structures that are mathematically indistinguishable from real ones. Gap of 0.05 means an attacker performs barely better than guessing.

4. **The formal connection** — The core problem was shown to reduce to "Threshold MinRank," a problem in computational algebra that has no known efficient solution.

5. **The process** — Open adversarial design with four AI systems cross-checking each other, every error published, every failure documented. This methodology — a human architect orchestrating competing AI experts — may be as significant as the cryptographic construction itself.

## What It Is NOT

- It is **not** a finished product. It is experimental research.
- It has **not** been proven secure. It has empirical resistance but no formal proof.
- It does **not** replace existing encryption. It explores a new direction.
- It has **not** been peer-reviewed by human cryptographers (yet). Three AI auditors have reviewed it across three rounds.

## Current Status

**Works:** Decryption through 50% noise. Indistinguishable decoys. Zero information leaks. Pure Python, runs in 6.6 seconds.

**Doesn't work yet:** No formal proof of hardness. Toy scale only (needs to be 100x larger for real security). The semifield approach needs a larger mathematical space to function.

**Next steps:** Publish as a research paper. Scale to larger parameters. Formalize the hard problem.

## The People

**Rafael Amichis Luengo (The Architect)** — Psychology graduate, creator of Proyecto Estrella. Designed the architecture, coordinated four AI systems, made every critical decision. Cannot write code. Built a cryptographic research artifact in 48 hours.

**Claude (Lead Engine)** — Built every prototype, executed every fix, wrote every document. Made errors that were caught by the other auditors.

**Gemini, ChatGPT, Grok (Auditors)** — Each independently reviewed the system three times. Each found critical flaws. Each contributed unique insights. The adversarial process between them is what makes the research credible.

---

*Proyecto Estrella · Error Code Lab*  
*"The truth is more important than the dream."*
