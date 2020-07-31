#!/usr/bin/env python3
"""
Exponential distribution
"""


class Exponential:
    """
    Class to represent a poisson distribution
    """
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        Exponential Constructor
        data is a list of the data to be used to estimate the distribution
        lambtha is the expected number of occurences in a given time frame
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = 1 / (float(sum(data) / len(data)))

    def pdf(self, x):
        """
        Probability density function
        Calculates the value of the PDF for a given time period
        x is the time period
        """
        if x < 0:
            return 0
        return self.lambtha * Exponential.e ** (-self.lambtha * x)

    def cdf(self, x):
        """
        Cumulative distribution function
        Calculates the value of the CDF for a given number of “successes”
        x is the time period
        """
        if x < 0:
            return 0
        return 1 - Exponential.e ** (-self.lambtha * x)
