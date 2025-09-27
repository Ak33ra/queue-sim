# display_system.py
from collections import defaultdict, deque
import math

def _server_label(s, idx):
    # prefer s.name if present, else class name, else index
    return getattr(s, "name", None) or s.__class__.__name__ or f"S{idx}"

def _build_edges(servers, P):
    n = len(servers)
    edges = []
    # detect explicit exit column
    has_exit_col = len(P[0]) == n + 1
    for i in range(n):
        row = P[i]
        upto = n if has_exit_col else len(row)
        for j in range(upto):
            p = row[j]
            if p and p > 0:
                edges.append((i, j, float(p)))
        if not has_exit_col:
            r = 1.0 - sum(row)
            if r > 1e-12:
                edges.append((i, "EXIT", r))
        else:
            p_exit = row[-1]
            if p_exit and p_exit > 0:
                edges.append((i, "EXIT", float(p_exit)))
    return edges

def display_system_ascii(servers, P):
    """ASCII, layer-ish layout via Kahn on the non-exit subgraph."""
    n = len(servers)
    labels = {i: _server_label(servers[i], i) for i in range(n)}
    edges = _build_edges(servers, P)

    # Build graph (excluding EXIT for layering)
    out = defaultdict(list)
    indeg = defaultdict(int, {i:0 for i in range(n)})
    for u,v,p in edges:
        if v == "EXIT":
            continue
        out[u].append((v,p))
        indeg[v] += 1

    # Kahn layering (if cycles, fall back to single layer)
    q = deque([i for i in range(n) if indeg[i] == 0])
    layers, seen = [], set()
    while q:
        layer = list(q)
        layers.append(layer)
        q.clear()
        for u in layer:
            seen.add(u)
            for v,_ in out[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
    if len(seen) != n:
        layers = [list(range(n))]  # cycle present → one row

    # Render
    print("\nQUEUEING NETWORK\n")
    colw = max(10, max(len(f"{i}:{labels[i]}") for i in range(n)) + 2)
    for L in layers:
        print("  ".join(f"[{i}:{labels[i]}]".ljust(colw) for i in L))
        # Edge lines under each layer’s nodes
        for u in L:
            outs = [(v,p) for v,p in out[u]]
            if outs:
                routes = ", ".join(
                    f"{u}→{v} ({p:.2f})" for v,p in outs
                )
                print("   " + routes)
        print()
    # Show exits
    exits = [(u,p) for (u,v,p) in edges if v == "EXIT"]
    if exits:
        print("Exits:")
        print("   " + ", ".join(f"{u}→EXIT ({p:.2f})" for u,p in exits))
    print()

def to_dot(servers, P):
    """Return Graphviz DOT string."""
    n = len(servers)
    labels = {i: _server_label(servers[i], i) for i in range(n)}
    edges = _build_edges(servers, P)

    lines = [
        "digraph G {",
        "  rankdir=LR;",
        '  node [shape=box, fontname="Menlo"];',
    ]
    for i in range(n):
        lines.append(f'  "{i}" [label="{i}: {labels[i]}"];')
    lines.append('  "EXIT" [shape=doublecircle, style=dashed];')
    for u,v,p in edges:
        vstr = v if v == "EXIT" else str(v)
        lines.append(f'  "{u}" -> "{vstr}" [label="{p:.2f}"];')
    lines.append("}")
    return "\n".join(lines)
