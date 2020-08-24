#!/usr/bin/env python3
"""
Optimization Module
"""
import numpy as np


def moving_average(data, beta):
    """
    Calculates the WMA using bias correction

    data is the list of data to calculate the moving average of
    beta is the weight used for the moving average

    Returns: a list containing the moving averages of data
    """
    vt = 0
    ema = []
    for i in range(len(data)):
        vt = (beta * vt) + ((1 - beta) * data[i])
        ema.append(vt / (1 - (beta ** (i + 1))))
    return ema
