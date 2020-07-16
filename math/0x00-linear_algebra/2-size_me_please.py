#!/usr/bin/env python3


def matrix_shape(matrix):
    shape = []
    return rec_matrix_shape(matrix, shape)


def rec_matrix_shape(matrix, shape):
    if type(matrix) == list:
        shape.append(len(matrix))
        rec_matrix_shape(matrix[0], shape)
    return shape
