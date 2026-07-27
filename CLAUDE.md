# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

GLoW (Gravitational Lensing of Waves) computes the GW amplification factor `F(w)` for wave-optics
lensing. Python package (`glow`) with a compiled C/Cython backend. GPL3.

## Build & install

```console
pip install .                       # full install (runs configure.py, builds C lib + Cython ext)
make                                # in-place dev build (equivalent to setup_wrapper below)
make -C wrapper                     # builds glow_lib/lib/libglow.a, then the .so extensions in-place
python3 configure.py -gsl <dir> -cc <compiler> -O 3 -cf <flag> -lf <flag>   # configure only
make clean                          # also restores the *.bak templates (see below)
```

`make` (target `setup_wrapper`) additionally creates the symlinks `notebooks/glow -> ..` and
`tests/glow -> ..`, which is what makes `from glow import ...` work from those directories without
installing. Anything run from `notebooks/` or `tests/` depends on those symlinks existing.

If a build fails, `configure.log` (repo root) has the details; `wrapper/.tmp_log` holds the
raw duplicated stdout.

### Generated build files — important

`configure.py` **rewrites** `wrapper/Makefile`, `wrapper/setup.py`, `wrapper/glow_lib/Makefile` and
`wrapper/tests/*/Makefile` in place, saving the pristine tracked version to `<file>.bak` on first
run. So after any build, those tracked files show as modified with machine-injected flags.

- Edit the `.bak` if present, never the generated file — the next configure run overwrites it.
- Never commit the configure-generated versions. `make clean` moves the `.bak`s back.
- `wrapper/tests/{test_gsl,test_omp,test_complex}` are configure-time capability probes (does GSL
  link? does OpenMP work? does C99 complex work?), not a test suite. If OpenMP probing fails the
  build silently falls back to serial mode.

Only GSL is a hard external dependency. `conda env create --file glow_env.yml` provides it.
Windows is unsupported (`configure.py` raises).

## Tests

```console
python3 tests/test_lenses.py                 # all lenses: derivatives + C/Python cross-check
make -C wrapper/glow_lib test                # build the C unit tests
./wrapper/glow_lib/tests/test_contour        # run one C test (binaries take no args)
make -C wrapper/glow_lib valgrind            # leak check
```

There is no pytest/CI setup. `tests/test_lenses.py` is a hand-rolled harness printing colored
PASS/FAIL: for each lens it checks analytic vs. finite-difference derivatives, the C implementation's
internal consistency, and Python-vs-C agreement. To test a single lens, edit the `lenses = [...]`
list in its `__main__` block. A lens whose `psi()` returns `None` is treated as Python-unimplemented
and skipped rather than failed; `check_implemented_lens(Psi)` decides whether the C path is tested.

## Docs

```console
make doc          # executes example notebooks -> html, then builds sphinx
make doc_only     # skip the (slow) notebook execution
make clean_nb     # strip notebook outputs/metadata before committing
```

`make doc_only` symlinks `sphinx_doc/glow -> ..` and runs `sphinx_doc/store_defaults.py`, which
dumps the live default `p_phys`/`p_prec` dicts so the docs stay in sync with the code — regenerate
docs after changing any default parameter dict.

## Architecture

The physics pipeline is `ψ(x) → I(τ) → F(w)`, and the class structure mirrors it exactly:

| Stage | Module | Base class |
|---|---|---|
| lensing potential `ψ(x)` | `lenses.py` | `PsiGeneral`, `PsiAxisym` |
| time-domain integral `I(τ)` | `time_domain.py` / `time_domain_c.py` | `ItGeneral` / `ItGeneral_C` |
| amplification factor `F(w)` | `freq_domain.py` / `freq_domain_c.py` | `FwGeneral` / `FwGeneral_C` |

Objects are chained by construction: `Fw_FFT_C(It_SingleContour_C(Psi_SIS(), y=1.2))`. Each stage
evaluates onto an internal grid and is then callable as an interpolator (`It(ts)`, `Fw(ws)`).

**Two parallel implementations.** The pure-Python modules and the `*_c.py` modules are independent
of each other (both depend only on `lenses.py`). The C versions supersede the Python ones in nearly
every respect and are what users should use; the Python ones exist as a dependency-free reference
and cross-check target. They do not offer identical subclass sets — e.g. `It_MultiContour_C` and
`Fw_DirectFT_C` have no Python counterpart. When adding a feature, decide deliberately whether it
lands in one or both; when adding a lens, adding it to both is what keeps `test_lenses.py`'s
cross-check meaningful.

**Parameter convention.** Every class takes two dicts, `p_phys` (physical) and `p_prec` (precision
/ algorithm control), each merged over a class-specific default dict. This is the pervasive idiom —
new options belong in these dicts, not as constructor keyword arguments. The C-side global
precision settings are separate, reached via `wrapper.get_Cprec / update_Cprec / display_Cprec`.

**Supporting modules.** `physical_units.py` converts between dimensionless code units and physical
ones; its `Units_*` classes are paired one-to-one with the `Psi_*` lenses (adding a lens generally
means adding a matching `Units_*`, dispatched by `Units()` / `Lens_Units()`). `waveform.py` builds
lensed/unlensed frequency- and time-domain waveforms and PSDs, and is the one place that pulls in
`pycbc` and detector sensitivity curves from `sensitivities/*.txt`.

### Compiled layer (`wrapper/`)

```
wrapper/glow_lib/source/*.c + include/*.h   ->  glow_lib/lib/libglow.a   (pure C, GSL + OpenMP)
wrapper/glow_lib/pocketfft/                     vendored FFT, compiled into libglow.a
wrapper/src/*.pyx + *.pxd                   ->  wrapper/c*.cpython-*.so  (Cython, links libglow.a)
wrapper/__init__.py                             re-exports the py* symbols from every .so
```

C module names map 1:1 onto Python-visible ones: `lenses_lib.c` → `clenses.pyx` → `clenses.so`,
similarly `roots`, `single_contour`, `contour`, `single_integral`, `area`, `special`, `fourier`,
`analytic_SIS`, `common`. Adding a C routine that should be callable from Python means touching the
`.h`/`.c`, the `.pxd` declaration, the `.pyx` wrapper, and the re-export in `wrapper/__init__.py`.

`wrapper/src/*.c` are Cython-generated and gitignored-by-convention (`make clean` deletes them) —
edit the `.pyx`, never the `.c`.

A new lens must be registered on the C side too (`lenses_lib.c` + its id in `clenses.pyx`) or
`check_implemented_lens` will report it as Python-only and the C path will be silently skipped.

## Other directories

- `scripts/glow_paper/` — reproduces every figure in the GLoW paper. `PL_approx_plot.py` takes
  ~1 hour on first run (mpmath) and caches its data afterwards.
- `notebooks/` — `examples_old.ipynb` is from the prerelease version; new tutorials are in progress.
