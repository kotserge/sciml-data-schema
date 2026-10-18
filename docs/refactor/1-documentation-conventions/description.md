# Refactor 1 — Documentation to the template's conventions

## Use case / Problem

`sciml-template` settled the documentation standard the SciML repositories share
(`sciml-template/CLAUDE.md`, 2026-10-15): numpydoc docstrings enforced by ruff's `D` and
`ANN` families, comments that describe, a README shaped Introduction / Installation / Usage /
Structure with a figure of the main idea, a `CLAUDE.md` with the conventions and the gotchas,
and rationale under `docs/`. This package predated it: prose docstrings carrying rationale, a
root `SCHEMA.md` marked "Draft" with chat residue and a layout tree behind the code (no
`registry_version`, `grid_type` or `provenance` attribute; per-field identity attributes the
writer does not store), no `CLAUDE.md`, no figure, untyped tests, and ruff without the `D`
and `ANN` families.

## Goal

- Every module, class and public function documented in numpydoc; ruff `D` (numpy
  convention, D105/D107/D401 off, `tests/*` skips D103) and `ANN` (ANN401 off) are gates,
  with `ruff format --check` and basedpyright over `src`, `tests` and `examples`.
- `README.md` in the template's shape; `CLAUDE.md` with the guiding idea, the layout, the
  gates, how to extend the schema, the conventions and the gotchas; a `justfile` with the
  gates.
- The specification moved to `docs/SCHEMA.md` and brought level with the code.
- `docs/figures/overview.py` draws the README's figure.

Non-goals: changes to the format or the API; the mesh profile; a version bump.

## Decisions

Note: `SCHEMA.md` moves to `docs/SCHEMA.md`. The root keeps the representing documents
(`README.md`, `CLAUDE.md`, `LICENSE`); the format and its rationale are `docs/` material.
Rationale that lived in docstrings and comments (why the registry is versioned separately,
why gzip level 1 with byte-shuffle, why the vocabularies are exported from the root) moved to
`docs/SCHEMA.md` and `CLAUDE.md`.

Note: the spec's inaccuracies are fixed, not documented around: the root attributes
`registry_version`, `grid_type` and `provenance`; the field datasets carry only the extent
and symmetry flags (identity comes from the registry); the time axis is a coordinate vector
(no `@dt`); the mesh profile is marked specified, not implemented.

Note: no code behaviour changed. The only edits beyond docstrings and comments are type
annotations (the `ANN` gate), module-level constants for the example's file name and field
names and for `io.py`'s `mask` / `values` dataset names, and the `import os` / `import h5py`
moved from function scope to module scope.

Note: `DatasetWriter._abort` unlinks `path` on any failure, so a pre-existing file opened
with `mode="a"` is deleted when the first `create_group` fails. Recorded as a gotcha in
`CLAUDE.md`; the fix (refuse any mode but `"w"`, or only delete a file the writer created)
is a follow-up, out of this pass's scope.
