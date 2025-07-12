import torch


def create_touching_edges(P):
    """
    Returns a list of directed edges (i -> j) for a PxP grid
    each node i connects with its 8 touching neighbors j
    """
    edges = []
    for row in range(P):
        for col in range(P):
            i = row*P + col  # Node index

            # Consider all the 8 neighbors (d_row, d_col)
            for d_row in (-1, 0, 1):
                for d_col in (-1, 0, 1):
                    # Skip (0,0) which is node i itself
                    if d_row == 0 and d_col == 0:
                        continue
                    n_row = row + d_row
                    n_col = col + d_col 

                    # Stay within grid boundary
                    if 0 <= n_row < P and 0 <= n_col < P:
                        j = n_row*P + n_col  # Neighbor index
                        edges.append((i,j))
    return edges



if __name__ == "__main__":
    # Smoke test for P = 3 and P = 4
    from collections import Counter

    for P in (3, 4):
        edges = create_touching_edges(P)
        print(f"P = {P}, nodes = {P*P}, directed edges = {len(edges)}")

        # Count how many outgoing edges each node has
        cnt = Counter(i for i, _ in edges)

        # Example nodes: corner (0), edge (1), center
        examples = {
            "corner": 0,
            "edge": 1,
            "center": (P//2)*P + (P//2)
        }
        for name, node in examples.items():
            print(f"  {name} node {node} has {cnt[node]} neighbors")
        print("  Sample edges:", edges[:8])
