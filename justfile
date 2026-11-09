# List the recipes
default:
    @just --list

# Install the package and dev dependencies
sync:
    uv sync --extra dev

# Format the code
fmt:
    uv run ruff format

# Lint the code and check its formatting
lint:
    uv run ruff check
    uv run ruff format --check

# Type-check the code
typecheck:
    uv run basedpyright

# Run the test suite
test *args:
    uv run pytest {{args}}

# Run every gate: lint, type check, tests
check: lint typecheck test
