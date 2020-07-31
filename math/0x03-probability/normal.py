#!/usr/bin/env python3
"""
Normal distribution
"""


class Normal:
    """
    Class to represent a Normal distribution
    """
    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Normal Constructor
        data is a list of the data to be used to estimate the distribution
        mean is the mean of the distribution
        stddev is the standard deviation of the distribution
        """
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = float(sum(data) / len(data))
            aux = 0
            for i in data:
                aux += (i - self.mean) ** 2
            self.stddev = (aux / len(data)) ** (1/2)

    @staticmethod
    def erf(x):
        a = 2 / (Normal.pi ** (1/2))
        b = (x - ((x ** 3) / 3) + ((x ** 5) / 10) -
             ((x ** 7) / 42) + ((x ** 9) / 216))
        return a * b

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value
        x is the x-value
        Returns the z-score of x
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score
        z is the z-score
        Returns the x-value of z
        """
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value
        x is the x-value
        Returns the PDF value for x
        """
        coef = 1 / (self.stddev * (2 * Normal.pi) ** (1/2))
        powe = -1/2 * (((x - self.mean) / self.stddev) ** 2)
        return coef * Normal.e ** powe

    def cdf(self, x):
        """
        Cumulative distribution function
        Calculates the value of the CDF for a given time period
        x is the time period
        """
        return (1/2) * (1 + Normal.erf(((x - self.mean) / (self.stddev
                                        * (2 ** (1/2))))))
