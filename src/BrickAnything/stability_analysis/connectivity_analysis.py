import networkx as nx
import numpy as np
from itertools import combinations


def connectivity_score(bricks) -> np.ndarray:
    """
    :param bricks: BrickStructure object representing the brick structure.
    :return: An array of voxels containing 0 if the voxel is connected to the ground via a series of brick connections,
             and 1 if it is not connected.
    """
    # Construct connectivity graph. Note that graph construction is O(N^2) in the number of bricks.
    graph = nx.Graph()
    graph.add_node('ground')
    graph.add_nodes_from(bricks.bricks)

    for b in bricks.bricks:  # Add edges for bricks connected to the ground
        if _connected_to_ground(b):
            graph.add_edge('ground', b)

    for b1, b2 in combinations(bricks.bricks, 2):  # Add edges for bricks connected to each other
        if _connected(b1, b2):
            graph.add_edge(b1, b2)

    # Find bricks connected to the ground
    connected_bricks = set(nx.node_connected_component(graph, 'ground'))

    result = np.zeros((bricks.world_dim, bricks.world_dim, bricks.world_dim))
    for brick in bricks.bricks:
        if brick not in connected_bricks:
            result[brick.slice] = 1

    return result


def _connected(b1, b2) -> bool:
    """
    Check if two bricks are connected.
    :param b1: First Brick object.
    :param b2: Second Brick object.
    :return: True if the bricks are connected (one on top of the other), False otherwise.
    """
    if b1.z > b2.z:
        b1, b2 = b2, b1

    if b1.z != b2.z - 1:
        return False

    # Check rectangle overlap in the x-y plane
    return (b1.x < b2.x + b2.h and b1.x + b1.h > b2.x and
            b1.y < b2.y + b2.w and b1.y + b1.w > b2.y)


def _connected_to_ground(b) -> bool:
    return b.z == 0
