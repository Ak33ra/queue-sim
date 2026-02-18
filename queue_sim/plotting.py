"""Convenience plotting functions for queueing analysis.

All functions return ``(fig, ax)`` so callers can further customize.
Requires matplotlib — install with ``pip install queue-sim[viz]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure


def _import_matplotlib():
    """Import matplotlib.pyplot, raising a helpful error if missing."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install queue-sim[viz]"
        ) from None
    return plt


def _ensure_ax(ax, plt):
    """Return (fig, ax), creating a new figure if ax is None."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    return fig, ax


def plot_cdf(
    response_times: Sequence[float],
    *,
    ax: matplotlib.axes.Axes | None = None,
    label: str | None = None,
    **kwargs,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the empirical CDF of response times.

    Args:
        response_times: Iterable of response time measurements.
        ax: Matplotlib axes to plot on. If None, a new figure is created.
        label: Legend label for this curve.
        **kwargs: Passed to ``ax.step()``.

    Returns:
        (fig, ax) tuple.
    """
    plt = _import_matplotlib()
    fig, ax = _ensure_ax(ax, plt)

    sorted_rt = np.sort(response_times)
    n = len(sorted_rt)
    cdf = np.arange(1, n + 1) / n

    ax.step(sorted_rt, cdf, where="post", label=label, **kwargs)
    ax.set_xlabel("Response time")
    ax.set_ylabel("P(T ≤ t)")
    ax.set_ylim(0, 1.05)
    if label is not None:
        ax.legend()

    return fig, ax


def plot_tail(
    response_times: Sequence[float],
    *,
    ax: matplotlib.axes.Axes | None = None,
    label: str | None = None,
    log: bool = True,
    **kwargs,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the empirical tail probability P(T > t).

    Args:
        response_times: Iterable of response time measurements.
        ax: Matplotlib axes to plot on. If None, a new figure is created.
        label: Legend label for this curve.
        log: If True (default), use log scale on the y-axis.
        **kwargs: Passed to ``ax.step()``.

    Returns:
        (fig, ax) tuple.
    """
    plt = _import_matplotlib()
    fig, ax = _ensure_ax(ax, plt)

    sorted_rt = np.sort(response_times)
    n = len(sorted_rt)
    tail = 1.0 - np.arange(1, n + 1) / n

    ax.step(sorted_rt, tail, where="post", label=label, **kwargs)
    ax.set_xlabel("Response time")
    ax.set_ylabel("P(T > t)")
    if log:
        ax.set_yscale("log")
    if label is not None:
        ax.legend()

    return fig, ax


def compare_policies(
    rt_dict: dict[str, Sequence[float]],
    *,
    kind: str = "cdf",
    ax: matplotlib.axes.Axes | None = None,
    **kwargs,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Overlay response time distributions for multiple policies.

    Args:
        rt_dict: Mapping of policy name to response times,
                 e.g. ``{"FCFS": rt1, "SRPT": rt2}``.
        kind: ``"cdf"`` (default) or ``"tail"``.
        ax: Matplotlib axes to plot on. If None, a new figure is created.
        **kwargs: Passed through to the underlying plot function.

    Returns:
        (fig, ax) tuple.
    """
    if kind not in ("cdf", "tail"):
        raise ValueError(f"kind must be 'cdf' or 'tail', got {kind!r}")

    plt = _import_matplotlib()
    fig, ax = _ensure_ax(ax, plt)

    plot_fn = plot_cdf if kind == "cdf" else plot_tail
    for name, rt in rt_dict.items():
        plot_fn(rt, ax=ax, label=name, **kwargs)

    ax.set_title(f"Response time {kind.upper()} by policy")
    ax.legend()

    return fig, ax


__all__ = ["plot_cdf", "plot_tail", "compare_policies"]
