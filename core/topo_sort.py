def topological_sort(edges) -> bool:
    g = {}
    for u, v in edges:
        if u not in g:
            g[u] = [v]
        else:
            g[u].append(v) 

    visited = set()
    cycle = set()

    def dfs(node):
        if node in visited:
            return True
        if node in cycle: 
            return False

        cycle.add(node)
        for nei in g.get(node, []):
            if not dfs(nei):
                return False 
        cycle.remove(node)
        visited.add(node)
        return True

    return all(dfs(node) for node in g)
