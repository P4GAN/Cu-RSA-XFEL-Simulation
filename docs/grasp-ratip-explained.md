# GRASP/RATIP: what it is and how this campaign works

A plain-language companion to `grasp/NEXT_SESSION.md` (the technical handoff for continuing this
work in a new session). This one is for you — what these tools are, why we're using them, what
we've found, and what's realistic to expect going forward.

## What is GRASP? What is RATIP?

Every atomic-physics number in this simulation's config files — how likely an X-ray photon is to
knock out a particular electron, how fast a particular hole decays, exactly what energy a
satellite line sits at — ultimately comes from solving the quantum mechanics of a copper atom
with some electrons missing. That's a genuinely hard many-electron problem, and there's no
shortcut around actually solving it numerically.

**GRASP** (General-purpose Relativistic Atomic Structure Package) does the first half: given a
description of an atomic state (which electrons are present, which are missing), it solves for
the electrons' wavefunctions and the state's total energy, including relativistic effects (which
matter a lot for an element as heavy as copper — a plain non-relativistic treatment would be
noticeably wrong). This is the same family of method — Dirac-Hartree-Fock — that essentially
every serious atomic structure calculation uses.

**RATIP** is a companion toolkit that takes GRASP's output and computes the things you actually
want from it: how strongly two states are coupled by a photon (giving transition rates and
photoionization cross sections), and how one state can decay into another by ejecting an
electron instead of a photon (Auger rates).

Both are real research-grade codes, but old — RATIP specifically dates to 2012 and hasn't been
actively maintained since. That matters practically: they're interactive command-line tools (you
answer a long sequence of prompts for each calculation) with no modern build tooling, and this
campaign has already hit and fixed one genuine bug in RATIP's 2012 source code (more on that
below).

## Why are we doing this instead of just using XATOM?

This project already had a way to compute these numbers: **XATOM**, a different atomic-structure
code (used via `xatom/xatom_tools.py`). The original paper this simulation is based on
(Chuchurka et al.) actually used GRASP + RATIP for its core numbers — XATOM was added later, for
extensions (satellite lines, the 2p₁/₂ pathway) that the original paper's model didn't have.

The problem: some of XATOM's numbers for those later extensions turned out to be wrong — not
just imprecise, but **wrong in sign** for a couple of the satellite-line detunings. That's the
kind of error that would make a simulated spectral feature show up on the wrong side of the main
peak instead of the right side. Having a second, independent method (GRASP/RATIP) that computes
the same physical quantities from scratch, using different mathematics and a different codebase,
is the standard way to catch that kind of mistake — if two independent calculations agree, that's
real evidence the number is right; if they disagree, at least one of them needs a closer look.

## How the process actually works

At a conceptual level, every calculation in this campaign follows the same four steps:

1. **Describe the atomic state.** Say precisely which electrons are present and which are
   missing — e.g. "a copper atom missing one 1s electron" (a K-shell hole), or "missing one 1s
   electron and also one 3d electron" (for a satellite line, where the 3d hole is a spectator
   sitting alongside the main K-hole).
2. **Solve for that state's wavefunctions and energy** (GRASP). This is an iterative
   self-consistent calculation — start from a rough guess, refine repeatedly until it converges.
3. **Compute the quantity you actually want** between two such states (RATIP): the energy
   difference (for a transition energy), the radiative transition rate (for a decay width or line
   strength), or the photoionization cross section (for how strongly the state absorbs a photon
   of a given energy).
4. **Convert units and compare** against the existing config value, to check agreement.

None of this is automated end-to-end — each step is its own command-line tool with its own
sequence of interactive prompts, and setting up a new atomic state (step 1) requires care to get
the quantum numbers and electron counts right. That's why this has been a slow, one-calculation-
at-a-time process rather than something that runs overnight unattended.

## What we've found so far

**The satellite-line detunings** (how far the "satellite" Kα lines — from atoms with a spectator
hole sitting in a 3d or 3p orbital during the Kα decay — sit away from the main line): all four
single-spectator channels now cross-checked against GRASP/RATIP, an independent third method
(JAC), and published literature. XATOM's numbers were genuinely wrong for two of the four — sign-
flipped, not just imprecise. One channel in particular (3p₋, a hole in the 3p₁/₂ orbital) turned
out to have a much bigger effect than XATOM predicted — its Kα2-satellite line sits about 4 eV
away from the main line, landing right around 8037 eV, which matches a feature you'd spotted by
eye in a published spectrum. That's a genuinely new, previously-missed result.

**Photoionization cross sections**: checked against essentially every value in the original
paper's own published table. Most of them now agree with GRASP/RATIP to well under 1% — about as
good an agreement as you could reasonably expect between two independent quantum-chemistry
calculations. A handful of the more complex ones (cross sections that are themselves a sum over
several possible outcomes) agree to a somewhat looser ~90%, which is normal for a summed quantity
where small errors in each piece add up.

One useful lesson from this part: a comparison that looked like a real 6× discrepancy turned out
to just be comparing the wrong things — a config value that predates a later model extension
covered a *broader* set of possibilities than the value I was computing. Once that scope was
matched properly, the agreement was essentially exact. Worth remembering if a future comparison
looks alarmingly off: check whether both sides are actually asking the same question before
concluding something is wrong.

## What's left, and what to expect

You asked to recompute *everything* — every base parameter and every satellite channel's full set
of values (not just detunings, but the decay rates, cross sections, and branching ratios too).
Realistically:

- **Every cross section and detuning value needed by the satellite-extended configs is now done.**
  These now live in a separate `-grasp` config variant, alongside the original XATOM-derived
  files, so both are available side by side.
- **Auger rates (decay rates and branching ratios) turned out not to need the debugging round we
  expected.** `xauger` — the one RATIP tool this campaign hadn't touched — worked cleanly on the
  first real attempt, no repeat of the bug-hunting the other tools needed. The genuinely hard part
  turned out to be a physics question instead: for some of these double-hole states, the two
  "flavors" of spectator hole (say, a hole in 3d₃/₂ vs 3d₅/₂) turn out to be so close in energy
  that the true quantum states are an inseparable mix of both, roughly half-and-half — a real
  effect, not a calculation error, and it needed its own explanation (`docs/auger-branching-and-
  ci-mixing.md`) and a specific way of splitting the rate between the two rather than picking one.
  That's now done for two of the four satellite channels; the other two are blocked by the same
  crash mentioned below.
- A couple of specific values hit unresolved technical snags (a stubborn convergence failure, an
  unexplained crash in one specific case) that were set aside rather than chased down, given the
  time already spent. Worth another attempt, but not guaranteed to resolve quickly. Total decay
  widths (how fast a hole state disappears via *every* available channel, not just the one
  dominant line already computed) are also still open.

None of this is because the method doesn't work — it clearly does, convincingly, for everything
tried so far. It's simply a lot of individual calculations, each with its own setup, and the
remaining categories are the ones that haven't been road-tested yet.

## A few practical things worth knowing

- **This is old, manual software.** Every calculation is a genuinely interactive back-and-forth
  with a 1990s/2000s-style command-line program, not a script you run and walk away from. Expect
  it to stay that way — there's no realistic way to fully automate this within the scope of this
  project.
- **Small setup mistakes tend to fail loudly** (a crash, a clearly wrong huge/negative number)
  rather than silently — which is reassuring, but does mean progress is punctuated by real
  troubleshooting, not steady linear progress.
- **A GRASP/RATIP number won't match a config value to the last decimal place**, and that's
  expected — it's an independent calculation, not a re-run of whatever produced the original
  number. Sub-1% agreement (seen for most values so far) is a strong result; 5-15% agreement for
  a more complex, multi-channel quantity is still a meaningful confirmation, not a red flag.
- **The absolute transition energies GRASP computes are consistently a few eV off from real
  experimental values** — this is normal and expected (GRASP without extremely expensive add-ons
  slightly underestimates some correlation effects). It's why every comparison in this campaign
  uses *relative* detunings anchored to the experimental Kα1/Kα2 energies, rather than trusting
  GRASP's own absolute energies directly.

If you want the deep technical detail — exact commands, every gotcha found, the full list of
what's left with pointers to where to start — that's all in `grasp/NEXT_SESSION.md` and the code
in `grasp/`.
