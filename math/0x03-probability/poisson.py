#!/usr/bin/python3
"""
Poisson distribution
"""


class Poisson:
    """
    Class to represent a poisson distribution
    """
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        Poisson Constructor
        data is a list of the data to be used to estimate the distribution
        lambtha is the expected number of occurences in a given time frame
        """
        self.lambtha = float(lambtha)
        if data is None:
            if lambtha < 0:
                raise ValueError('lambtha must be a positive value')
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = float(sum(data) / len(data))
