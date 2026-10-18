# CLAUDE.md

Guidance for Claude Code in this repository.

## What this repo is

The **HDF5 dataset format and metadata schema** the SciML repositories (`sciml-template` and
its forks) exchange simulation data through: a data-generation workspace writes files in this
format, a training workspace reads them, and neither imports the other. One uv project, one
package (`sciml_data_schema`), two runtime dependencies (`h5py`, `numpy`). The format is
modeled on The Well's and specified in `docs/SCHEMA.md`; the code is its reference
implementation.

## The guiding idea

One file is one dataset of `n_trajectories` runs over a shared domain. Every physical quantity
is a `Field` with an *identity* shared across files (tensor order, role, unit: its
`field_registry` entry) and an *extent* per file (the `dim_varying` / `time_varying` /
`sample_varying` flags: the axes its array carries). The stored shape follows from the flags
and `Dataset.validate` checks it. The role tag is what a consumer routes on: `state` to the
prediction targets, everything else to the conditioning inputs.

```
src/sciml_data_schema/
├── schema.py        # the objects and the canonical HDF5 names; validate(); imports no h5py
├── io.py            # the only module that imports h5py: write_dataset, DatasetWriter, read_dataset
├── provenance.py    # git_revision, for the provenance stamp
└── __init__.py      # the public API, re-exported
```

The layering is strict: `schema.py` knows nothing of files, `io.py` serializes the schema
objects and nothing else, and every name the file uses is a constant in `schema.py`.

## Layout

- `src/sciml_data_schema/` — the package; `__init__.py` re-exports the whole public API.
- `tests/` — pytest, offline. `conftest.py` builds one fixture with a field of every tensor
  order and extent, which the round-trip, validation and writer tests mutate per case;
  `test_example.py` runs the example at toy size.
- `examples/advection_diffusion.py` — a complete producer/consumer scenario on the public API.
- `docs/SCHEMA.md` — the format; `docs/figures/overview.py` draws the README's figure;
  `docs/<category>/<N>-<slug>/` is the planning trail (see *Conventions*).
- `justfile` — `sync`, `test`, `check`, `format`, `figure`, `example`.

## Gates

```bash
uv run pytest -q
uv run ruff check src tests examples docs/figures
uv run ruff format --check src tests examples docs/figures
uv run basedpyright
```

`just test` and `just check` run them. ruff enforces pycodestyle, pyflakes, isort, pyupgrade,
bugbear, pydocstyle (the numpy convention; D401, D105 and D107 are off, `tests/*` skips D103)
and flake8-annotations; `ruff format` is the formatter. `basedpyright` runs in `standard` mode
over `src`, `tests` and `examples`; never pipe it (a pipe swallows its exit code). The figure
script is linted, not type-checked (matplotlib is not in the environment; uv resolves it from
the script's inline metadata).

## Extending the schema

- **A root attribute or group:** its name as an `ATTR_*` / `GROUP_*` constant in `schema.py`,
  the field on `Dataset`, the writer and the reader in `io.py` (read with a default, so older
  files still load), a round-trip test, the layout tree in `docs/SCHEMA.md` §4. Bump
  `SCHEMA_VERSION` when the layout changes.
- **A per-field attribute:** the same path through `Field`, `_write_fields`, `_read_fields`,
  and the field table in `docs/SCHEMA.md` §1.
- **A vocabulary entry** (a role, a BC type, a representation): the `Literal` in `schema.py`
  (and `REPRESENTATIONS` for a representation), `docs/SCHEMA.md` §1 / §3. Bump
  `REGISTRY_VERSION` when the canonical vocabulary changes.
- **A profile** (the mesh profile's `geometry/` group, `docs/SCHEMA.md` §6): its groups as
  constants, its writer and reader, validation of its arrays, a fixture and round-trip tests,
  the profile section of the spec.

## Conventions

- **Names: full words.** `n_trajectories`, `normalization`, `boundary_conditions`. Python
  identifiers and HDF5 names are snake_case; dataset files are kebab-case
  (`poisson-validation.h5`).
- **Canonical names are constants.** Every HDF5 group, attribute and JSON key is spelled once,
  in `schema.py`; `io.py` never writes a literal name.
- **Validate before touching disk.** Both writers call `Dataset.validate()` first and never
  leave a partial file; the reader validates what it loaded.
- **Public surface.** Everything a producer or a consumer needs is importable from
  `sciml_data_schema`; the `Literal` vocabularies (`Role`, `Representation`, `GridType`,
  `BCType`) are exported because they appear in the public signatures.
- **Three version numbers.** `SCHEMA_VERSION` (the layout), `REGISTRY_VERSION` (the
  vocabulary) and the package version (`pyproject.toml`, `__version__`) move independently
  (`docs/SCHEMA.md` §10).
- **Documentation:** comments and docstrings state what the code does; the README states what
  a user needs; the format and its rationale live in `docs/SCHEMA.md`, decisions in
  `docs/<category>/…` (`~/.claude/CLAUDE.md`).
- **Docstrings** are numpydoc (every module, class and public function; ruff's `D` rules with
  `convention = "numpy"` enforce presence and shape). The summary is one line stating what the
  unit computes or holds, as a noun phrase or an imperative. Then the sections, each present
  when it applies: `Attributes` (a dataclass's fields as `name : type`), `Parameters` (every
  argument as `name : type`, with its shape), `Returns` (type, then what it is), `Raises` (the
  exception, then the condition), `Notes` (the one place for a hidden assumption a caller must
  know), `Examples` (a literal block). A docstring never says why a design was chosen, what it
  replaced, or that it is safe, fast or careful. Borrowed code carries one terse attribution
  line in `Notes`. The shape:

  ```python
  def expected_ndim(self, n_spatial_dims: int) -> int:
      """The number of array axes the extent flags and the tensor order imply.

      Parameters
      ----------
      n_spatial_dims : int
          The dataset's number of spatial dimensions.

      Returns
      -------
      int
          One axis per set leading flag, ``n_spatial_dims`` axes when ``dim_varying``,
          plus ``tensor_order`` trailing axes.
      """
  ```
- **README** tells a reader what the package is and how to use it: Introduction (with the
  figure), Installation, Usage (with the API table), Structure. Not the format's rationale,
  not open points; those are `docs/SCHEMA.md`.
- **Planning.** Non-trivial work goes through `docs/<category>/<N>-<slug>/` (description →
  research → plan, reviewed between stages; `Note:` lines in the files are the decisions).
  Categories: `feature`, `bug`, `refactor`, `test`, `migration`, `idea`, `review`.
- **Commits:** Conventional Commits (`type(scope): summary`); scopes `schema`, `io`, `docs`,
  `tests`; one step per commit.

## Gotchas

- **`Normalization` statistics are arrays.** Build them with `np.asarray(field.data.mean())`:
  a bare `.mean()` is a `float64`, not the `np.ndarray` the type says, and basedpyright flags
  the consumer.
- **`read_dataset` is eager:** every field is loaded whole. A loader over files larger than
  memory reads the `t*_fields` datasets with `h5py` directly, by the names in `schema.py`.
- **`field_placeholder` is a read-only broadcast view;** writing into it raises. It only stands
  in for a streamed field's shape and dtype.
- **`DatasetWriter` deletes `path` on any failure,** including a pre-existing file opened with
  `mode="a"` or `"r+"` (the first `create_group` on an existing group fails and the abort
  unlinks the path). The writer is for new files.
- **A global constant has two spellings,** a 0-d array or `(1,)`. Both validate; the reader
  returns whichever was written.
- **h5py attribute values** come back as `bytes` for legacy strings and as numpy scalars for
  bools and ints; `io.py`'s `_as_*` helpers normalize them, so never read an attribute raw.
- **Streamed fields must be `sample_varying`:** the leading axis is what `append` indexes.
- **Chunking** applies to streamed fields only (one trajectory, or one trajectory × timestep,
  per chunk); an inline field takes h5py's default layout.
- `requires-python` is 3.11, so no `type X = ...` aliases and no PEP 695 generics, even though
  the venv may run a newer interpreter.
- `ruff format` reflows hand-wrapped code: run it before a docstring or comment pass, not after.
