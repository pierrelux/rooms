"""Dynamic programming methods for policy evaluation and control
"""

import numpy as np


def solve_mdp(P, R, gamma, policies):
    """ Batch Policy Evaluation Solver

    We denote by 'A' the number of actions, 'S' for the number of
    states, 'N' for the number of policies evaluated and 'K' for the
    number of reward functions to evaluate.

    Args:
      P (numpy.ndarray): Transition function as (A x S x S) tensor
      R (numpy.ndarray): Reward function as a (S x A x K) tensor
      gamma (float): Scalar discount factor
      policies (numpy.ndarray): tensor of shape (N x S x A)

    Returns:
      tuple (vfs, qfs) where the first element is a tensor of shape
      (N x S X K) and the second element contains the Q functions as a
      tensor of shape (N x S x A x K).
    """
    nstates = P.shape[-1]
    ppi = np.einsum('ast,nsa->nst', P, policies)
    rpi = np.einsum('sak,nsa->nsk', R, policies)
    vfs = np.linalg.solve(np.eye(nstates) - gamma*ppi, rpi)
    qfs = R + gamma*np.einsum('ast,ntk->nsak', P, vfs)
    return vfs, qfs


def value_iteration(P, R, gamma, num_iters=10):
    """Value iteration for the Bellman optimality equations

    Args:
        P (np.ndarray): Transition function as (A x S x S) tensor
        R (np.ndarray): Reward function as a (S x A) matrix
        gamma (float): Discount factor
        num_iters (int, optional): Defaults to 10. Number of iterations

    Returns:
        tuple: value function and state-action value function tuple
    """
    nstates, nactions = P.shape[-1], P.shape[0]
    qf = np.zeros((nstates, nactions))
    for _ in range(num_iters):
        qf = R + gamma*np.einsum('ast,t->sa', P, np.max(qf, axis=1))
    return np.max(qf, axis=1), qf
