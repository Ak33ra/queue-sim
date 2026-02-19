"""Network animation for queueing simulation trajectories.

Requires matplotlib. For smart layout, ``pip install networkx``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.animation
    import matplotlib.figure


def _import_deps():
    """Import matplotlib and return (plt, animation)."""
    try:
        import matplotlib.animation as anim
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for animation. "
            "Install it with: pip install queue-sim[viz]"
        ) from None
    return plt, anim


def _infer_edges(log) -> set[tuple[int, int]]:
    """Infer directed edges from observed ROUTE events in the log."""
    edges: set[tuple[int, int]] = set()
    for i in range(len(log)):
        if log.kinds[i] == "route":
            fr = log.from_servers[i]
            to = log.to_servers[i]
            if fr >= 0 and to >= 0:
                edges.add((fr, to))
    return edges


def _layout_positions(
    n_servers: int,
    edges: set[tuple[int, int]],
    positions: dict[int, tuple[float, float]] | None,
    transition_matrix: list[list[float]] | None,
) -> dict[int, tuple[float, float]]:
    """Compute node positions for the network graph."""
    if positions is not None:
        return positions

    # Try networkx spring layout
    try:
        import networkx as nx

        G = nx.DiGraph()
        G.add_nodes_from(range(n_servers))
        if transition_matrix is not None:
            for i, row in enumerate(transition_matrix):
                for j in range(min(len(row), n_servers)):
                    if row[j] > 0:
                        G.add_edge(i, j)
        else:
            G.add_edges_from(edges)
        return nx.spring_layout(G, seed=42)
    except ImportError:
        pass

    # Fallback: check if it's a tandem (linear chain)
    is_tandem = all(
        (i, i + 1) in edges for i in range(n_servers - 1)
    ) and len(edges) == n_servers - 1

    if is_tandem or n_servers <= 2:
        # Linear layout
        return {i: (i / max(n_servers - 1, 1), 0.5) for i in range(n_servers)}

    # Circular layout
    pos = {}
    for i in range(n_servers):
        angle = 2 * np.pi * i / n_servers
        pos[i] = (0.5 + 0.4 * np.cos(angle), 0.5 + 0.4 * np.sin(angle))
    return pos


def animate_network(
    log,
    n_servers: int | None = None,
    *,
    transition_matrix: list[list[float]] | None = None,
    positions: dict[int, tuple[float, float]] | None = None,
    n_frames: int = 300,
    interval: int = 50,
    cmap: str = "YlOrRd",
    figsize: tuple[float, float] = (10, 7),
    node_size: float = 800,
    show_colorbar: bool = True,
    title: str | None = None,
) -> matplotlib.animation.FuncAnimation:
    """Animate the network state over time from an event log.

    Args:
        log: An EventLog (Python or C++) with event data.
        n_servers: Number of servers (inferred if None).
        transition_matrix: Optional routing matrix for edge drawing.
        positions: Optional node positions ``{server_idx: (x, y)}``.
        n_frames: Number of animation frames.
        interval: Milliseconds between frames.
        cmap: Colormap for node coloring by occupancy.
        figsize: Figure size.
        node_size: Size of server nodes.
        show_colorbar: Whether to show a colorbar.
        title: Optional figure title.

    Returns:
        ``matplotlib.animation.FuncAnimation`` — call ``.save()`` or display
        inline in Jupyter.
    """
    from .event_log import _bin_step_function, per_server_states

    plt, animation_mod = _import_deps()

    data = per_server_states(log, n_servers=n_servers)
    n_srv = len(data["server_states"])
    times_arr = np.array(data["times"])

    t_max = times_arr[-1] if len(times_arr) > 0 else 1.0
    bin_edges = np.linspace(0, t_max, n_frames + 1)

    # Bin each server's occupancy
    grids = np.zeros((n_srv, n_frames))
    for s in range(n_srv):
        grids[s] = _bin_step_function(
            times_arr, data["server_states"][s], bin_edges
        )

    vmax = max(float(grids.max()), 1.0)
    colormap = plt.get_cmap(cmap)
    norm = plt.Normalize(vmin=0, vmax=vmax)

    # Infer edges for drawing
    edges = _infer_edges(log)
    if transition_matrix is not None:
        for i, row in enumerate(transition_matrix):
            for j in range(min(len(row), n_srv)):
                if row[j] > 0:
                    edges.add((i, j))

    pos = _layout_positions(n_srv, edges, positions, transition_matrix)

    # Set up figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")

    if title:
        ax.set_title(title)

    # Draw edges (static)
    for fr, to in edges:
        x0, y0 = pos[fr]
        x1, y1 = pos[to]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={"arrowstyle": "->", "color": "gray", "lw": 1.5},
        )

    # Draw external arrival arrow into server 0
    if 0 in pos:
        x0, y0 = pos[0]
        ax.annotate(
            "",
            xy=(x0, y0),
            xytext=(x0 - 0.15, y0),
            arrowprops={"arrowstyle": "->", "color": "steelblue", "lw": 2},
        )
        ax.text(x0 - 0.17, y0, "arr", ha="right", va="center", fontsize=8)

    # Draw departure arrows from exit servers
    exit_servers = set()
    for i in range(len(log)):
        if log.kinds[i] == "departure":
            fr = log.from_servers[i]
            if fr >= 0:
                exit_servers.add(fr)
    for s in exit_servers:
        if s in pos:
            x0, y0 = pos[s]
            ax.annotate(
                "",
                xy=(x0 + 0.15, y0),
                xytext=(x0, y0),
                arrowprops={"arrowstyle": "->", "color": "indianred", "lw": 2},
            )

    # Server node circles (initial)
    node_xs = [pos[s][0] for s in range(n_srv)]
    node_ys = [pos[s][1] for s in range(n_srv)]
    initial_colors = [colormap(norm(grids[s, 0])) for s in range(n_srv)]
    scatter = ax.scatter(
        node_xs, node_ys, s=node_size, c=initial_colors,
        edgecolors="black", linewidths=1.5, zorder=5,
    )

    # Server ID labels (below the node)
    for s in range(n_srv):
        ax.text(
            pos[s][0], pos[s][1] - 0.08, f"S{s}",
            ha="center", va="top", fontsize=9, color="0.3", zorder=6,
        )

    # Occupancy labels (on the node, updated per frame)
    occ_texts = []
    for s in range(n_srv):
        t = ax.text(
            pos[s][0], pos[s][1], f"{grids[s, 0]:.0f}",
            ha="center", va="center", fontsize=12, fontweight="bold",
            color="white", zorder=7,
        )
        occ_texts.append(t)

    # Time counter
    time_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes,
        ha="left", va="top", fontsize=10,
    )

    # Colorbar
    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Occupancy", shrink=0.6)

    def _update(frame):
        colors = [colormap(norm(grids[s, frame])) for s in range(n_srv)]
        scatter.set_facecolors(colors)
        for s in range(n_srv):
            val = grids[s, frame]
            occ_texts[s].set_text(f"{val:.0f}")
            # Dark text on bright nodes, white on dark nodes
            occ_texts[s].set_color("black" if val / vmax > 0.5 else "white")
        t = (bin_edges[frame] + bin_edges[frame + 1]) / 2
        time_text.set_text(f"t = {t:.2f}")
        return (scatter, time_text, *occ_texts)

    anim = animation_mod.FuncAnimation(
        fig, _update, frames=n_frames, interval=interval, blit=True,
    )

    return anim


__all__ = ["animate_network"]
