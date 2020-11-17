#!/usr/bin/env python3
"""
Markov Module
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden markov model:

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

    Returns: P, F, or None, None on failure
        P is the likelihood of the observations given the model
        F is a numpy.ndarray of shape (N, T) containing the forward path
        probabilities
            F[i, j] is the probability of being in hidden state i at time j
            given the previous observations
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
    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    for t in range(1, Observation.shape[0]):
        F[:, t] = (F[:, t - 1].dot(Transition[:, :])) * \
            Emission[:, Observation[t]]
    P = np.sum(F[:, -1])
    return (P, F)


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden markov model:

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

    Returns: P, B, or None, None on failure
        Pis the likelihood of the observations given the model
        B is a numpy.ndarray of shape (N, T) containing the backward path
        probabilities
            B[i, j] is the probability of generating the future observations
            from hidden state i at time j
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
    if Transition.shape != (Emission.shape[0], Emission.shape[0]):
        return None, None
    if Initial.shape != (Emission.shape[0], 1):
        return None, None

    B = np.zeros((Emission.shape[0], Observation.shape[0]))
    B[:, Observation.shape[0] - 1] += 1
    for t in range(Observation.shape[0] - 2, -1, -1):
        B[:, t] = (B[:, t + 1] * (Transition[:, :])
                   ).dot(Emission[:, Observation[t + 1]])
    P = np.sum(B[:, 0] * Initial.T * Emission[:, Observation[0]])
    return (P, B)


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a hidden markov model:

    Observations is a numpy.ndarray of shape (T,) that contains the index of
    the observation
        T is the number of observations
    Transition is a numpy.ndarray of shape (M, M) that contains the
    initialized transition probabilities
        M is the number of hidden states
    Emission is a numpy.ndarray of shape (M, N) that contains the initialized
    emission probabilities
        N is the number of output states
    Initial is a numpy.ndarray of shape (M, 1) that contains the initialized
    starting probabilities
    iterations is the number of times expectation-maximization should be
    performed

    Returns: the converged Transition, Emission, or None, None on failure
    """
    N, _ = Transition.shape
    T = Observations.shape[0]
    for i in range(iterations):
        P1, alpha = forward(Observations, Emission, Transition, Initial)
        P2, beta = backward(Observations, Emission, Transition, Initial)
        xi = np.zeros((N, N, T - 1))
        for t in range(T - 1):
            ems = Emission[:, Observations[t + 1]].T
            den = np.dot(np.multiply(np.dot(alpha[:, t].T, Transition), ems),
                         beta[:, t + 1])
            for i in range(N):
                a = Transition[i]
                num = alpha[i, t] * a * ems * beta[:, t + 1].T
                xi[i, :, t] = num / den
        gamma = np.sum(xi, axis=1)
        Transition = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2],
                                         axis=0).reshape((-1, 1))))
        den = np.sum(gamma, axis=1)
        for i in range(Emission.shape[1]):
            Emission[:, i] = np.sum(gamma[:, Observations == i], axis=1)
        Emission = np.divide(Emission, den.reshape((-1, 1)))
    return Transition, Emission
