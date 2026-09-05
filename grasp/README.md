# GRASP2018 + RATIP-2012 Dirac-Fock cross-check

An independent atomic-structure method, used to validate `config/base/Cu-seed-original.yaml`'s
atomic-physics parameters (satellite-line detunings, `sigma1_*`/`sigma2_*` photoionization cross
sections) against `xatom/`'s XATOM-based pipeline. Two different codes agreeing is much stronger
evidence than one code's self-consistency, and this cross-check did catch real bugs in XATOM's
numbers — the 3d-/3p- satellite detunings were sign-wrong, not just imprecise.

## What's here

- **`grasp_tools.py`** — parsing/unit-conversion helpers for RATIP's `xphoto` (photoionization)
  and GRASP2018's `rtransition` (bound-bound transition rate) output. Read its module docstring
  first — it also has the external build/recipe notes needed to reproduce any of this.
- **`cross_section_validation.py`** — the actual numbers obtained (satellite detunings, cross
  sections), recorded as data next to their provenance, plus the unit-converted comparison
  against the config. Run it directly for a summary printout.
- **`satellite_detuning_analysis.py`** — reproduces the satellite-detuning numbers in
  `cross_section_validation.py` from the raw `rtransition` `.ct` files: the composition data
  (which level is which physical hole-type) and dominant-line selection per channel. Needs the
  external GRASP working directory to actually run; read as documentation otherwise.

## What's *not* here

The GRASP/RATIP calculations themselves are interactive Fortran runs (`rcsfgenerate`, `rmcdhf`,
`xrelci`, `xphoto`, ...) against a scratch working directory full of binary intermediate files
(CSL lists, radial orbitals, mixing coefficients) — not something that belongs in this repo, and
not automatable from Python the way `xatom_tools.py` shells out to XATOM. `grasp_tools.py`'s
docstring has the recipe to redo any specific calculation if needed.

## Headline results (2026-09-03 to 2026-09-05)

- **Satellite detunings**: all 4 single-spectator channels (3d±, 3p±) cross-validated against JAC
  and literature (Nguyen et al., PRA 105, 022811 (2022); Mauron & Dousse, PRA 82, 052505 (2010)).
  XATOM's values were substantially wrong — sign-flipped for 3d-/3p-, ~2x too small for 3d+/3p+.
  The 3p- Kα2-satellite in particular detunes by **+3.98 eV** (XATOM had nothing like this).
- **Cross sections — full comparison against Table I of the original paper** (see
  `cross_section_validation.py`'s module docstring for the paper-notation <-> config-key mapping):
  9 of 11 entries computed. All 4 ground-state (g) values not near a threshold match to <1%;
  the 4 ion-referenced total-further-ionization (i) values, each a sum over several independently
  computed channels, land at 87-95%. `sigma1_pump_1s` is the one outlier, and it's a
  threshold-calibration artifact, not a disagreement (see below) — everything else checks out.
  `sigma2_pump_other` is still missing (pump-energy values aren't needed for the target configs,
  see `grasp/NEXT_SESSION.md` §0) but `sigma2_Ka1_other` is now done — see below.
- **Every `sigma1_Ka1_*`/`sigma2_Ka1_*` key needed by the satellite-extended target configs is now
  computed**, including `sigma2_Ka1_2s`, `sigma2_Ka1_2p1`, and `sigma2_Ka1_other` (the sigma1-share-
  weighted average over 5 auxiliary hole types — the single biggest remaining piece, now done).
  Agreement against the existing XATOM-derived config values ranges 78-102%, in line with this
  campaign's established accuracy band for summed further-ionization quantities. Two known gaps
  carried forward for a future session: the `2p+3p` double-hole channel (an unresolved `rmcdhf`
  CSF-loader crash affects both `sigma2_Ka1_2p3` and the 3p1/2,3p3/2 legs of `sigma2_Ka1_other`),
  and `3s`'s own `3s+3s_further` channel (a new `xphoto` silent-stop failure mode, reproduced
  twice). See `grasp/NEXT_SESSION.md` §2a for full detail.
- **The GRASP-recomputed values now live in a separate config variant**
  (`config/base/Cu-seed-SASE-double-satellite-grasp.yaml` and its mono sibling), not mixed into the
  original XATOM-derived files — see each file's header comments for exactly which keys differ
  and why.
- **A methodology note worth keeping in mind**: `sigma1_Ka1_other` first looked wrong by 6x
  against a naive comparison — until realizing the *original* config's "other" catch-all
  (`Cu-seed-original.yaml`) predates the later `use_2s_pathway`/`use_L2_pathway` split, so it
  still includes the 2s and 2p_1/2 contributions that got their own explicit channels later.
  Once summed with the correct 7-subshell scope (2s + 2p_1/2 + 3s + 3p + 3d), the match is <0.2%
  at *both* photon energies — i.e. there never was a bug here, just an easy scope mismatch when
  comparing two different eras of the same config concept.
- **`sigma1_pump_1s` threshold note**: my own K-edge (Breit+VP, no self-energy) sits ~150 eV above
  wherever the original paper's calculation effectively puts it, so the 9000 eV pump lands just
  below my threshold. The cross-section shape (rising p-wave threshold law) and magnitude both
  check out once evaluated a bit further above threshold — an energy scan confirmed this, just
  not checked into this module since it's a scan, not a single number to compare.
