"""One neutral, low-fatigue visual style shared by every TM-Scan figure."""
from contextlib import contextmanager
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# Ink and structure
INK, MUTED, FAINT, GRID, SURFACE = "#222222", "#5A5A58", "#9A9995", "#E3E2DE", "#FFFFFF"
TOTAL = "#444444"      # sums over families (F1+F2 ...)
ACCENT = "#9C3D33"     # rejected / flagged items; never a family color

# Families take these colors by ORDER of appearance, whatever their names.
FAMILY_COLORS = ("#245B83", "#B05A3C", "#5B95A6", "#6B4C8A", "#A6851F", "#2F6B4F")

# Orientation models keep one color each in every figure.
METHOD_COLORS = {"observed": INK, "fisher": "#245B83", "kent": "#B05A3C", "bingham": "#2F6B4F",
                 "spherical_kde": "#6B4C8A", "empirical": "#A6851F"}
METHOD_LABELS = {"observed": "Measured", "fisher": "Fisher", "kent": "Kent", "bingham": "Bingham",
                 "spherical_kde": "Spherical KDE", "empirical": "Empirical"}

# Sequential maps: one hue, light -> dark (magnitude). Ordinal steps for radii.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "tmscan_sequential", ["#F2F1EC", "#C9D9DF", "#8FB4C4", "#4F87A3", "#245B83", "#12324B"])
# Same hue for point maps: starts darker so low values stay visible on white.
SEQUENTIAL_POINTS = LinearSegmentedColormap.from_list(
    "tmscan_points", ["#D3DEE2", "#A9C4CF", "#6F9DB5", "#3E7395", "#245B83", "#12324B"])
SEQUENTIAL_WARM = LinearSegmentedColormap.from_list(
    "tmscan_sequential_warm", ["#F5F1EA", "#E6CDB4", "#D09A78", "#B05A3C", "#6E2F1E"])
for _cmap in (SEQUENTIAL, SEQUENTIAL_POINTS, SEQUENTIAL_WARM):
    try:
        mpl.colormaps.register(_cmap)
    except ValueError:      # already registered (module reloaded)
        pass

STYLE = {
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.7,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "image.cmap": "tmscan_sequential",
    "axes.prop_cycle": mpl.cycler(color=FAMILY_COLORS),
}


def family_colors(families):
    """{family: color} by order; the 7th and later families reuse the muted ink."""
    names = list(families)
    return {name: (FAMILY_COLORS[i] if i < len(FAMILY_COLORS) else MUTED) for i, name in enumerate(names)}


def ordinal_colors(count, cmap=None):
    """Ordered light->dark steps (e.g. one per window radius)."""
    cmap = cmap or SEQUENTIAL
    if count <= 1:
        return [cmap(0.75)]
    return [cmap(0.25 + 0.7 * i / (count - 1)) for i in range(count)]


def use_style():
    """Apply the TM-Scan style to every later matplotlib figure of the session."""
    mpl.rcParams.update(STYLE)


_KEEP = ("backend", "interactive")


@contextmanager
def style_context(extra=None):
    """Temporary TM-Scan style; `extra` overrides individual rcParams.

    Unlike matplotlib.rc_context, the backend and interactive mode are not
    restored on exit: Jupyter's inline backend switches interactive mode on
    when it loads (on the first figure), and undoing that would stop every
    later figure from being displayed.
    """
    context = mpl.rc_context({**STYLE, **(extra or {})})
    context.__enter__()
    try:
        yield
    finally:
        kept = {key: dict.__getitem__(mpl.rcParams, key) for key in _KEEP}
        context.__exit__(None, None, None)
        dict.update(mpl.rcParams, kept)
