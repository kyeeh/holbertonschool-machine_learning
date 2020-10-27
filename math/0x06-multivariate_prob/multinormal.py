#!/usr/bin/env python3
"""
Multivariate Probability
"""
import numpy as np


class MultiNormal():
    """
    Multivariate Normal distribution
    """

    def __init__(self, data):
        """
        Constructor method

        data is a numpy.ndarray of shape (d, n) containing the data set:
        n is the number of data points
        d is the number of dimensions in each data point
        If data is not a 2D numpy.ndarray, raise a TypeError with the message
        data must be a 2D numpy.ndarray
        If n is less than 2, raise a ValueError with the message data must
        contain multiple data points
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if n < 2:
            raise ValueError("data must contain multiple data points")
        d = data.shape[0]
        n = data.shape[1]            
        self.mean = np.mean(data, axis=1).reshape(d, 1)
        cvrc = data - self.mean
        self.cov = np.dot(cvrc, cvrc.T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the PDF at a data point

        x is a numpy.ndarray of shape (d, 1) containing the data point whose
        PDF should be calculated
        d is the number of dimensions of the Multinomial instance

        If x is not a numpy.ndarray, raise a TypeError with the message x must
        be a numpy.ndarray
        If x is not of shape (d, 1), raise a ValueError with the message x
        must have the shape ({d}, 1)

        Returns the value of the PDF
        """
        d = self.cov.shape[0]
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if (len(x.shape) != 2 or x.shape[1] != 1 or x.shape[0] != d):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        x_hat = x - self.mean
        pdf = (1 / (np.sqrt((2 * np.pi)**d * np.linalg.det(self.cov)))
               * np.exp(-(np.linalg.solve(self.cov, x_hat).T.dot(x_hat)) / 2))

        return float(pdf)
