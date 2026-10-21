"""Shared matplotlib style: fixed colour per model (never cycled), recessive axes."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# Categorical slots from the validated reference palette, assigned per entity.
COLORS = {
    "GLM": "#2a78d6", "CANN": "#eb6834", "LightGBM": "#1baf7a", "FFNN (embeddings)": "#eda100",
    "FFNN (one-hot)": "#e87ba4", "Homogeneous": "#8a8984", "Actual": INK,
}
SEQ = ["#cfe0f6", "#9ec2ee", "#6aa1e3", "#2a78d6", "#1d5aa6", "#123d73"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "lines.linewidth": 2, "figure.dpi": 110, "savefig.dpi": 150,
    "savefig.bbox": "tight",
})
