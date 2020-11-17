#!/usr/bin/env python3
"""
Markov Module
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates the most likely sequence of hidden states for a hidden markov
    model:

    Observation is a numpy.ndarray of shape (T,) that contains the index of
    the observation
        T is the number of observations
    Emission is a numpy.ndarray of shape (N, M) containing the emission
    probability of a specific observation given a hidden state
        Emission[i, j] is the probability of observing j given the hidden
        state i
        N is the number of hidden states
        M is the number of all possible observations
    Transition is a 2D numpy.ndarray of shape (N, N) containing the transition
    probabilities
        Transition[i, j] is the probability of transitioning from the hidden
        state i to j
    Initial a numpy.ndarray of shape (N, 1) containing the probability of
    starting in a particular hidden state

    Returns: path, P, or None, None on failure
        path is the a list of length T containing the most likely sequence of
        hidden states
        P is the probability of obtaining the path sequence
    """
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None
    if Transition.shape != (Emission.shape[0], Emission.shape[0]):
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None
    if Initial.shape != (Emission.shape[0], 1):
        return None, None
    if not np.sum(Emission, axis=1).all():
        return None, None
    if not np.sum(Transition, axis=1).all():
        return None, None
    if not np.sum(Initial) == 1:
        return None, None

    F = np.zeros((Emission.shape[0], Observation.shape[0]))
    backpointer = np.zeros((Emission.shape[0], Observation.shape[0]))

    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    backpointer[:, 0] = 0

    for t in range(1, Observation.shape[0]):
        for s in range(Emission.shape[0]):
            F[s, t] = np.max(F[:, t - 1] * Transition[:, s] *
                             Emission[s, Observation[t]])
            backpointer[s, t] = np.argmax(F[:, t - 1] * Transition[:, s] *
                                          Emission[s, Observation[t]])
    path = [0] * Observation.shape[0]
    path[-1] = np.argmax(F[:, Observation.shape[0] - 1])
    for t in range(Observation.shape[0] - 1, 0, -1):
        path[t - 1] = int(backpointer[path[t], t])
    P = np.max(F[:, -1])
    return path, P
