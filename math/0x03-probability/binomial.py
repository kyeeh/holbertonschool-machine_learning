#!/usr/bin/env python3
"""
Binomial distribution
"""


class Binomial:
    """
    Class to represent a Binomial distribution
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Binomial Constructor
        data is a list of the data to be used to estimate the distribution
        n is the number of Bernoulli trials
        p is the probability of a “success”
        """
        if data is None:
            if n <= 0:
                raise ValueError('n must be a positive value')
            if p <= 0 or p >= 1:
                raise ValueError('p must be greater than 0 and less than 1')
            self.n = round(n)
            self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = float((sum(data)) / len(data))
            var = 0
            for i in data:
                var += ((mean - i) ** 2)
            var = var / len(data)
            p = 1 - (var / mean)
            self.n = round(mean / p)
            self.p = float(mean / self.n)

    @staticmethod
    def factorial(n):
        """
        Calculates the factorial of n
        """
        fn = 1
        for i in range(2, n + 1):
            fn *= i
        return fn

    @staticmethod
    def combinatory(n, r):
        """
        Calculates combinatory of n in r.
        """
        return Binomial.factorial(n) / (Binomial.factorial(r)
                                        * Binomial.factorial(n - r))

    def pmf(self, k):
        """
        Probability mass function
        Calculates the value of the PMF for a given number of “successes”
        k is the number of “successes”
        """
        k = int(k)
        if k < 0:
            return 0
        cnk = Binomial.combinatory(self.n, k)
        return cnk * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """
        Cumulative distribution function
        Calculates the value of the CDF for a given number of “successes”
        k is the number of “successes”
        """
        k = int(k)
        if k < 0:
            return 0
        acumulated = 0
        for i in range(k + 1):
            acumulated += self.pmf(i)
        return acumulated
