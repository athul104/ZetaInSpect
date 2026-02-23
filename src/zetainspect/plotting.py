from __future__ import annotations
from typing import TYPE_CHECKING, Iterable, Literal, Optional, Tuple, Union
import numpy as np
from .utils import asarray_numeric
from .scales import Scales

if TYPE_CHECKING:
    from .background import InflationHistory
    from .spectrum import Spectrum

__all__ = [
    "plot_quantity",
    "add_top_N_axis",
]


# -----------------------------
# Small helpers
# -----------------------------

def _get_history(obj) -> "InflationHistory":
    """Accept InflationHistory or Spectrum and return the underlying InflationHistory."""
    if hasattr(obj, "epsilon_H") and hasattr(obj, "Hubble") and hasattr(obj, "eta"):
        # likely InflationHistory
        return obj
    if hasattr(obj, "history"):
        return obj.history
    raise TypeError("Object must be InflationHistory or Spectrum-like with .history")

def _get_spectrum(obj) -> "Spectrum":
    """Require a Spectrum object for vk/Pzeta plotting."""
    if hasattr(obj, "vk") and hasattr(obj, "power_spectrum"):
        return obj
    raise TypeError("This plot requires a Spectrum object (with .vk/.power_spectrum methods).")

def _default_N_grid(history: "InflationHistory", n: int = 2000) -> np.ndarray:
    """Default N grid for plotting (keep moderate to avoid heavy Hankel costs)."""
    return np.linspace(history.N_max, history.N_min, int(n), dtype=history.dtype)

def _as_1d(x, dtype) -> np.ndarray:
    x = asarray_numeric(x, dtype=dtype)
    return np.asarray(x).reshape(-1)

def _vk_to_real(vk: np.ndarray, component: str) -> np.ndarray:
    if component == "abs":
        return np.abs(vk)
    if component == "real":
        return vk.real
    if component == "imag":
        return vk.imag
    raise ValueError("vk_component must be one of {'abs','real','imag'}")


# -----------------------------
# Top axis helper
# -----------------------------

def add_top_N_axis(
    ax,
    *,
    scales: Scales,
    ratio: float,
    N_ticks: Optional[np.ndarray] = None,
    fmt: str = "{:g}",
    xlabel: str = "N",
):
    """
    Add a top x-axis labeled by N on a plot whose bottom x-axis is k (ratio-mode).

    Parameters
    ----------
    ax : matplotlib Axes
        Existing axis with k on the x-axis.
    scales : Scales
        Scale converter.
    ratio : float
        Same horizon ratio used for k(N) on the bottom axis.
    N_ticks : array-like, optional
        Tick locations in N. If None, uses a coarse default.
    fmt : str, default "{:g}"
        Formatting for tick labels.
    xlabel : str
        Label for the top axis.

    Returns
    -------
    ax_top : matplotlib Axes
    """
    import matplotlib.pyplot as plt  # local import

    if N_ticks is None:
        # coarse defaults, user can override
        N_ticks = np.arange(5, 66, 5, dtype=float)

    N_ticks = np.asarray(N_ticks, dtype=float)
    k_ticks = np.asarray(scales.k_at_horizon_ratio(N_ticks, ratio), dtype=float).reshape(-1)

    # keep only ticks within current k limits
    kmin, kmax = ax.get_xlim()
    m = (k_ticks >= min(kmin, kmax)) & (k_ticks <= max(kmin, kmax))
    k_ticks = k_ticks[m]
    N_ticks = N_ticks[m]

    ax_top = ax.twiny()
    ax_top.set_xscale(ax.get_xscale())
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(k_ticks)
    ax_top.set_xticklabels([fmt.format(v) for v in N_ticks])
    ax_top.set_xlabel(xlabel)
    return ax_top


# -----------------------------
# Main dispatcher
# -----------------------------

XMode = Literal["N", "k"]
YMode = Literal["eta", "abs_eta", "epsilon", "H", "vk", "Pzeta"]
SpecMode = Literal["ratio", "track"]
VKComp = Literal["abs", "real", "imag"]

def plot_quantity(
    obj,
    *,
    y: YMode,
    x: XMode = "k",
    mode: SpecMode = "ratio",
    N: Optional[np.ndarray] = None,
    ratio: float = 0.005,
    k_fixed: Optional[float] = None,
    vk_component: VKComp = "abs",
    overlays: Iterable[str] = (),
    top_axis_N: bool = False,
    N_ticks: Optional[np.ndarray] = None,
    ax=None,
    logx: Optional[bool] = None,
    logy: Optional[bool] = None,
    label: Optional[str] = None,
    **plot_kwargs,
) -> Tuple["object", "object"]:
    """
    Plot a chosen quantity versus N or k.

    Parameters
    ----------
    obj : InflationHistory or Spectrum
        - For y in {'eta','epsilon','H'}: InflationHistory is enough.
        - For y in {'vk','Pzeta'}: must be Spectrum.
    y : {'eta', 'abs_eta', 'epsilon','H','vk','Pzeta'}
        Quantity to plot.
    x : {'N','k'}, default 'k'
        X-axis variable. Note: x='k' is supported only in mode='ratio'.
    mode : {'ratio','track'}, default 'ratio'
        - 'ratio': k is chosen as k(N, ratio); supports x='N' or x='k'
        - 'track': fixed k_fixed, supports x='N' only
    N : array-like, optional
        N grid. If None, a default grid is created.
    ratio : float, default 0.005
        Horizon ratio for ratio-mode plotting.
    k_fixed : float, optional
        Required when mode='track'.
    vk_component : {'abs','real','imag'}, default 'abs'
        How to turn complex vk into real values for plotting.
    overlays : iterable of str
        Any of: 'pivot', 'transitions', 'As', 'CMB', 'PBH'
        (CMB/PBH are placeholders; customize their numeric values.)
    top_axis_N : bool, default False
        If True and x='k' and mode='ratio', add a top axis labeled by N.
    N_ticks : array-like, optional
        Tick values for the top N axis.
    ax : matplotlib Axes, optional
        If provided, draw on it. Otherwise create a new fig/ax.
    logx, logy : bool, optional
        If None, defaults are chosen based on x/y.
    label : str, optional
        Legend label.
    plot_kwargs : dict
        Passed to ax.plot/ax.loglog etc. (lw, ls, alpha, etc.)

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    import matplotlib.pyplot as plt  # local import

    history = _get_history(obj)
    scales = getattr(obj, "scales", None)
    if scales is None:
        scales = Scales(history)

    if N is None:
        N_arr = _default_N_grid(history, n=2000)
    else:
        N_arr = asarray_numeric(N, dtype=history.dtype)

    if mode == "track":
        if k_fixed is None:
            raise ValueError("mode='track' requires k_fixed.")
        if x != "N":
            raise ValueError("mode='track' supports only x='N' (k is fixed).")

    if x == "k" and mode != "ratio":
        raise ValueError("x='k' is supported only for mode='ratio'.")

    # -------------------------
    # Compute y(N) first
    # -------------------------
    if y == "eta":
        yN = history.eta_H(N_arr)
    elif y == "abs_eta":
        yN = np.abs(history.eta_H(N_arr))
    elif y == "epsilon":
        yN = history.epsilon_H(N_arr)
    elif y == "H":
        yN = history.Hubble(N_arr)
    elif y in ("vk", "Pzeta"):
        spec = _get_spectrum(obj)
        if mode == "ratio":
            if y == "vk":
                vk = spec.vk(N_arr, ratio=ratio)
                yN = _vk_to_real(vk, vk_component)
            else:
                yN = spec.power_spectrum(N_arr, ratio=ratio)
        else:  # track
            if y == "vk":
                vk = spec.vk_track(N_arr, k_fixed)
                yN = _vk_to_real(vk, vk_component)
            else:
                yN = spec.power_spectrum_track(N_arr, k_fixed)
    else:
        raise ValueError("Unknown y quantity.")

    # -------------------------
    # Build x-values
    # -------------------------
    if x == "N":
        xvals = N_arr
        xlabel = "N"
    else:
        # ratio-mode mapping: k depends on N
        xvals = scales.k_at_horizon_ratio(N_arr, ratio)
        xlabel = r"$k\ \mathrm{(Mpc^{-1})}$"

    # -------------------------
    # Choose default log scales
    # -------------------------
    if logx is None:
        logx = (x == "k")
    if logy is None:
        # typical defaults: epsilon/H/Pzeta are positive; eta can be signed
        if y in ("epsilon", "H", "Pzeta"):
            logy = True
        elif y == "vk":
            logy = True if vk_component == "abs" else False
        else:
            logy = False

    # -------------------------
    # Create fig/ax if needed
    # -------------------------
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # -------------------------
    # Plot
    # -------------------------
    if label is None:
        label = y

    if logx and logy:
        ax.loglog(xvals, yN, label=label, **plot_kwargs)
    elif logx and not logy:
        ax.semilogx(xvals, yN, label=label, **plot_kwargs)
    elif (not logx) and logy:
        ax.semilogy(xvals, yN, label=label, **plot_kwargs)
    else:
        ax.plot(xvals, yN, label=label, **plot_kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(y)
    ax.set_xlim(min(xvals), max(xvals))

    # -------------------------
    # Overlays
    # -------------------------
    overlays = set(overlays)

    # Pivot marker
    if "pivot" in overlays:
        if x == "N":
            ax.axvline(history.N_pivot, color = 'purple', linestyle="--", alpha=0.8)
        else:
            # In ratio-mode, the k-value corresponding to N_pivot at the chosen ratio
            k_piv = float(history.k_pivot)
            ax.axvline(k_piv, color = 'purple', linestyle="--", alpha=0.8)

    # Transition markers
    if "transitions" in overlays and history.transitions.size > 0:
        if x == "N":
            for Nt in history.transitions:
                ax.axvline(float(Nt), linestyle="--", alpha=0.6, color='orange')
        else:
            for Nt in history.transitions:
                kt = float(scales.k_horizon_exit(float(Nt)))
                ax.axvline(kt, linestyle="--", alpha=0.6)

    # A_s horizontal line (only meaningful for Pzeta)
    if "As" in overlays and y == "Pzeta":
        ax.axhline(float(history.A_s), linestyle=":", alpha=0.8, color='darkgreen')

    # CMB window (placeholder defaults; adjust as needed)
    if "CMB" in overlays and x == "k":
        # Typical rough k-range for CMB analyses; customize to your convention.
        kmin_cmb = 0.0005
        kmax_cmb = 0.5
        ax.axvspan(kmin_cmb, kmax_cmb, alpha=0.1, color='lightblue')

    # PBH constraints (placeholder hook)
    if "PBH" in overlays and y == "Pzeta":
        ax.axhline(1e-2, color='red', linestyle='--', lw=1.4)

    # -------------------------
    # Optional top axis in N (for k-plots)
    # -------------------------
    if top_axis_N and (x == "k") and (mode == "ratio"):
        add_top_N_axis(ax, scales=scales, ratio=ratio, N_ticks=N_ticks, xlabel="N")

    # Legend only if user wants labels
    if label is not None:
        ax.legend()

    return fig, ax