"""Create grid world environments from string layouts
"""

import dp
import numpy as np
import matplotlib.pyplot as plt


def layout_to_array(layout, wall='w'):
    """Convert string layout to array representation

    Args:
        layout (str): Multi-line string where each line is a row of the grid world
        wall (str, optional): Defaults to 'w'. Character describing a wall

    Returns:
        np.ndarray: Numpy array (np.float) where a wall is represented by 0 and 1 for an empty cell.
    """

    return np.array([list(map(lambda c: 0 if c == 'w' else 1, line))
                     for line in layout.splitlines()])


def make_adjacency(layout):
    """Convert a grid world layout to an adjacency matrix.

    Args:
        layout (np.ndarray): Grid layout as an array where 0 means a wall and 1 is empty.

    Returns:
        tuple: First element is aulti-dimensional np.ndarray of size (A X S X S) where A=4 is the 
        number of actions, and S is the number of states. The action set is: 
        UP (0), DOWN (1), LEFT (2), RIGHT (3). The second element of the tuple is a np.ndarray
        mapping state (integer) to cell coordinates in the original layout.
    """
    directions = [np.array((-1, 0)),  # UP
                  np.array((1, 0)),  # DOWN
                  np.array((0, -1)),  # LEFT
                  np.array((0, 1))]  # RIGHT

    grid = layout_to_array(layout)
    state_to_grid_cell = np.argwhere(grid)
    grid_cell_to_state = {tuple(state_to_grid_cell[s].tolist()): s
                          for s in range(state_to_grid_cell.shape[0])}

    nstates = state_to_grid_cell.shape[0]
    nactions = len(directions)
    P = np.zeros((nactions, nstates, nstates))
    for state, idx in enumerate(state_to_grid_cell):
        for action, d in enumerate(directions):
            if grid[tuple(idx + d)]:
                dest_state = grid_cell_to_state[tuple(idx + d)]
                P[action, state, dest_state] = 1.

    return P, state_to_grid_cell


class FourRooms:
    """Deterministic four-rooms layout with sparse reward upon reaching the goal
    """

    def __init__(self):
        self.layout = """wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        self.P, _ = make_adjacency(self.layout)
        self.R = np.copy(np.swapaxes(self.P[:, :, -1], 0, 1))
        self.P[:, -1, :] = 0.
        self.P[:, -1, -1] = 1.
        self.discount = 0.99
        self.mdp = [self.P, self.R, self.discount]


def plot_policy(ax, layout, mdp):
    """Solve the MDP and plot the greedy policy on the original grid layout.

    Args:
        ax (matplotlib.pyplot.axis): Axis object for the given figure
        layout (str): Multi-line string where each line is a row of the grid world
        mdp (tuple): (P, R, gamma) where P is (A x S x S), R is (S x A) and gamma is a float [0,1)
    """

    grid = layout_to_array(layout)
    _, state_to_grid_cell = make_adjacency(layout)
    ax.imshow(grid)

    action_symbols = ['↑', '↓', '←', '→']
    _, qf = dp.value_iteration(*mdp, num_iters=50)
    for s, c in enumerate(state_to_grid_cell):
        symb = action_symbols[np.argmax(qf[s, :])]
        plt.text(c[1], c[0], symb, color='red', ha='center', va='center')

    gc = state_to_grid_cell[-1]
    ax.text(gc[1], gc[0], 'g')


if __name__ == "__main__":
    rooms = FourRooms()

    fig, ax = plt.subplots()
    plot_policy(ax, rooms.layout, rooms.mdp)

    plt.show()
