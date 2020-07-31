#!/usr/bin/env python3
"""
Poisson distribution
"""


class Poisson:
    """
    Class to represent a Poisson distribution
    """
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        Poisson Constructor
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
            self.lambtha = float(sum(data) / len(data))

    @staticmethod
    def factorial(n):
        """
        Calculates the factorial of n
        """
        fn = 1
        for i in range(2, n + 1):
            fn *= i
        return fn

    def pmf(self, k):
        """
        Probability mass function
        Calculates the value of the PMF for a given number of “successes”
        k is the number of “successes”
        """
        k = int(k)
        if k < 0:
            return 0
        return Poisson.e ** -self.lambtha * self.lambtha ** k \
            / Poisson.factorial(k)

    def cdf(self, k):
        """
        Cumulative distribution function
        Calculates the value of the CDF for a given number of “successes”
        k is the number of “successes”
        """
        k = int(k)
        if k < 0:
            return 0
        return Poisson.e ** -self.lambtha * \
            sum([(self.lambtha ** i) / Poisson.factorial(i)
                 for i in range(k + 1)])
