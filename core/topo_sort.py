from graphlib import CycleError, TopologicalSorter


def topological_sort(edges) -> bool:
    graph = TopologicalSorter()
    for winner, loser in edges:
        graph.add(loser, winner)

    try:
        graph.prepare()
    except CycleError:
        return False
    return True
