# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8"]
# ///
"""Draw the overview figure of the root README, ``docs/figures/overview.png``.

The top row is the data flow: a producer builds a ``Dataset`` and writes it (``write_dataset``,
or ``DatasetWriter`` trajectory by trajectory), the file holds the shared core layout, and a
consumer reads it back (``read_dataset``) and routes the fields by role. The bottom panel is the
field model: a field's identity (the registry entry: tensor order, role, unit), its extent (the
three flags), the stored shape they imply, and four examples.

Examples
--------
::

    uv run docs/figures/overview.py

Notes
-----
Layout and palette adapted from ``sciml-template/docs/figures/overview.py`` and
``sciml-template/analysis/scripts/_style.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "overview.png"
DPI = 200

# The neutrals, the two hues of the code boxes, and the fill of the file box
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
PRODUCER = "#2a78d6"
CONSUMER = "#1baf7a"
FILE_EDGE = "#8f8e8a"
FILE_FILL = "#f1f0ed"

# Geometry in inches: a row of three boxes above one wide panel
MARGIN_X = 0.45
BOX_GAP = 0.7
ROW_BOTTOM = 4.35
ROW_HEIGHT = 2.9
PANEL_BOTTOM = 0.45
PANEL_HEIGHT = 3.1
TEXT_PAD = 0.22
TITLE_SIZE = 12.5
BODY_SIZE = 8.4
SMALL_SIZE = 7.8
LINE_SPACING = 1.32
ROUNDING = "round,pad=0,rounding_size=0.14"
ARROW_STYLE = "-|>"


@dataclass(frozen=True)
class Box:
    """One top-row box: a code area (producer, consumer) or the file between them.

    Attributes
    ----------
    title : str
        The bold title line.
    body : str
        The body lines, newline-separated, left-aligned in monospace.
    width : float
        The box width in inches.
    color : str | None
        The code area's hue, or ``None`` for the file box.
    """

    title: str
    body: str
    width: float
    color: str | None


BOXES = (
    Box(
        "producer",
        "dataset = Dataset(\n"
        "    dataset_name, representation,\n"
        "    n_spatial_dims, n_trajectories,\n"
        "    spatial_dims, dimensions, fields,\n"
        "    boundary_conditions, normalization,\n"
        "    provenance)\n"
        "\n"
        "write_dataset(dataset, path)\n"
        "\n"
        "with DatasetWriter(dataset, path,\n"
        '                   streamed=("u",)) as w:\n'
        '    w.append("u", trajectory, data)',
        3.6,
        PRODUCER,
    ),
    Box(
        "<dataset_name>.h5",
        "@schema_version   @registry_version\n"
        "@dataset_name     @representation   @grid_type\n"
        "@n_spatial_dims   @n_trajectories\n"
        "@field_registry   @present_fields   @provenance\n"
        "\n"
        "dimensions/           @spatial_dims   time, x, y\n"
        "boundary_conditions/  <name>/  @bc_type  mask  values\n"
        "normalization/        <field>/ @mean @std @min @max\n"
        "t0_fields/ t1_fields/ t2_fields/\n"
        "    <field>  @dim_varying @time_varying\n"
        "             @sample_varying",
        4.6,
        None,
    ),
    Box(
        "consumer",
        "dataset = read_dataset(path)\n"
        "\n"
        "route by role:\n"
        "  state        -> prediction targets\n"
        "  forcing      -> conditioning inputs\n"
        "  coefficient  -> conditioning inputs\n"
        "  parameter    -> conditioning vector\n"
        "\n"
        "standardise with the shipped\n"
        "normalization statistics",
        3.6,
        CONSUMER,
    ),
)
ARROW_LABELS = ("write", "read")

# The figure is exactly as wide as the row of boxes plus the margins
FIGURE_SIZE = (
    2 * MARGIN_X + sum(box.width for box in BOXES) + (len(BOXES) - 1) * BOX_GAP,
    ROW_BOTTOM + ROW_HEIGHT + 0.35,
)

PANEL_TITLE = "A field: an identity shared across files, an extent per file"
PANEL_IDENTITY = (
    "identity -- the field_registry entry\n"
    "  tensor_order  0 | 1 | 2   -> t0_fields | t1_fields | t2_fields,\n"
    "                               and the trailing (D,) * order axes\n"
    "  role          state | forcing | coefficient | parameter | boundary\n"
    '  unit          "m/s", "Pa", "dimensionless", ...'
)
PANEL_EXTENT = (
    "extent -- the axes this file's array carries\n"
    "  sample_varying  -> a leading (n_trajectories,) axis\n"
    "  time_varying    -> an (n_steps,) axis\n"
    "  dim_varying     -> the (X, Y[, Z]) axes\n"
    "  an axis whose flag is False is not stored"
)
PANEL_FORMULA = "stored shape = (n_trajectories?, n_steps?, *spatial_shape?, *(D,) * tensor_order)"
PANEL_EXAMPLES = (
    "u(x, y, t)   state        t0   dim yes  time yes  sample yes   (n_traj, n_steps, X, Y)\n"
    "velocity     state        t1   dim yes  time yes  sample yes   (n_traj, n_steps, X, Y, 2)\n"
    "kappa(x, y)  coefficient  t0   dim yes  time no   sample yes   (n_traj, X, Y)\n"
    "Re           parameter    t0   dim no   time no   sample yes   (n_traj,)"
)


def _box_x(index: int) -> float:
    """The left edge of box ``index`` on the top row."""
    return MARGIN_X + sum(box.width for box in BOXES[:index]) + index * BOX_GAP


def _draw_box(ax: Axes, index: int, box: Box) -> None:
    """A rounded box with a centred bold title and a left-aligned monospace body."""
    x = _box_x(index)
    is_code = box.color is not None
    ax.add_patch(
        FancyBboxPatch(
            (x, ROW_BOTTOM),
            box.width,
            ROW_HEIGHT,
            boxstyle=ROUNDING,
            linewidth=1.6,
            edgecolor=box.color if is_code else FILE_EDGE,
            facecolor=box.color if is_code else FILE_FILL,
        )
    )
    ax.text(
        x + box.width / 2,
        ROW_BOTTOM + ROW_HEIGHT - 0.3,
        box.title,
        ha="center",
        va="center",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        color=SURFACE if is_code else INK,
        family="monospace",
    )
    ax.text(
        x + TEXT_PAD,
        ROW_BOTTOM + ROW_HEIGHT - 0.62,
        box.body,
        ha="left",
        va="top",
        fontsize=BODY_SIZE,
        color=SURFACE if is_code else INK_MUTED,
        family="monospace",
        linespacing=LINE_SPACING,
    )


def _draw_arrow(ax: Axes, index: int, label: str) -> None:
    """The arrow from box ``index`` to the next one, with ``label`` above it."""
    x0 = _box_x(index) + BOXES[index].width
    x1 = _box_x(index + 1)
    y = ROW_BOTTOM + ROW_HEIGHT / 2
    ax.add_patch(
        FancyArrowPatch(
            (x0 + 0.05, y),
            (x1 - 0.05, y),
            arrowstyle=ARROW_STYLE,
            mutation_scale=16,
            linewidth=1.6,
            color=INK_MUTED,
        )
    )
    ax.text(
        (x0 + x1) / 2,
        y + 0.12,
        label,
        ha="center",
        va="bottom",
        fontsize=BODY_SIZE,
        color=INK_MUTED,
        family="monospace",
    )


def _draw_panel(ax: Axes) -> None:
    """The field-model panel: the identity and extent columns, the shape formula, the examples."""
    x0 = MARGIN_X
    width = FIGURE_SIZE[0] - 2 * MARGIN_X
    top = PANEL_BOTTOM + PANEL_HEIGHT
    ax.add_patch(
        FancyBboxPatch(
            (x0, PANEL_BOTTOM),
            width,
            PANEL_HEIGHT,
            boxstyle=ROUNDING,
            linewidth=1.4,
            edgecolor=FILE_EDGE,
            facecolor=SURFACE,
        )
    )

    # The title, then the two columns side by side
    ax.text(
        x0 + TEXT_PAD,
        top - 0.28,
        PANEL_TITLE,
        ha="left",
        va="center",
        fontsize=TITLE_SIZE - 1.5,
        fontweight="bold",
        color=INK,
        family="monospace",
    )
    column_top = top - 0.6
    for column_x, text in ((x0 + TEXT_PAD, PANEL_IDENTITY), (x0 + width / 2 + 0.3, PANEL_EXTENT)):
        ax.text(
            column_x,
            column_top,
            text,
            ha="left",
            va="top",
            fontsize=BODY_SIZE,
            color=INK_MUTED,
            family="monospace",
            linespacing=LINE_SPACING,
        )

    # The shape rule, centred, and the examples under it
    ax.text(
        x0 + width / 2,
        PANEL_BOTTOM + 1.22,
        PANEL_FORMULA,
        ha="center",
        va="center",
        fontsize=BODY_SIZE + 0.6,
        color=INK,
        family="monospace",
    )
    ax.text(
        x0 + TEXT_PAD,
        PANEL_BOTTOM + 0.98,
        PANEL_EXAMPLES,
        ha="left",
        va="top",
        fontsize=SMALL_SIZE,
        color=INK_MUTED,
        family="monospace",
        linespacing=LINE_SPACING,
    )


def main() -> None:
    """Draw the overview on the object API, write it to ``OUT`` and print the path."""
    # Axes over the whole figure, so one data unit is one inch and the geometry above holds
    figure = Figure(figsize=FIGURE_SIZE)
    ax = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    figure.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, FIGURE_SIZE[0])
    ax.set_ylim(0, FIGURE_SIZE[1])
    ax.set_aspect("equal")
    ax.axis("off")

    # The data flow: the three boxes and the two arrows between them
    for index, box in enumerate(BOXES):
        _draw_box(ax, index, box)
    for index, label in enumerate(ARROW_LABELS):
        _draw_arrow(ax, index, label)

    # The field model
    _draw_panel(ax)

    figure.savefig(OUT, dpi=DPI, facecolor=SURFACE, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
