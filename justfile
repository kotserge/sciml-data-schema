# Entry points of the schema package: one uv project, one environment. Needs `just`
# (https://github.com/casey/just) and uv; every recipe is a thin wrapper over `uv run`.

# Recipe arguments reach the recipe as "$@", one word each
set positional-arguments

# Show the available recipes.
default:
    @just --list

# Sync the environment with the dev extra (pytest, ruff, basedpyright).
sync:
    uv sync --extra dev

# Run the test suite (offline).
test:
    uv run pytest -q

# Lint, format-check and type-check the package, the tests, the example and the figure script.
check:
    uv run ruff check src tests examples docs/figures
    uv run ruff format --check src tests examples docs/figures
    uv run basedpyright

# Reformat in place.
format:
    uv run ruff format src tests examples docs/figures

# Redraw the README's overview figure (docs/figures/overview.png).
figure:
    uv run docs/figures/overview.py

# Run the advection-diffusion example; writes advection_diffusion_1d.h5 in the current directory.
example:
    uv run examples/advection_diffusion.py
