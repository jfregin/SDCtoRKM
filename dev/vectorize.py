#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 17:18:03 2022

Utility functions for vector computations
"""
import numpy as np


def matVecMul(mat, u):
    r"""
    Compute vectorized Matrix Vector Multiplication :math:`Ax` (A @ x)

    Parameters
    ----------
    mat : np.ndarray, size (nDOF, M, M) or (M, M)
        Matrix or array of matrices.
    u : np.ndarray, size (nDOF, M) or (M,)
        Vector or array of vectors.

    Returns
    -------
    out : np.ndarray, size (nDOF, M) or (M,)
        The computed matrix-vector product(s)

    Notes
    -----
    - matVecMul((nDOF, M, M), (nDOF, M)) -> (nDOF, M) <=> (M, M) @ (M,) for each nDOF
    - matVecMul((M, M), (nDOF, M)) -> (nDOF, M) <=> (M, M) @ (M,) for each nDOF
    - matVecMul((M, M), (M,)) -> (M,) <=> (M, M) @ (M,)
    """
    return np.matmul(mat, u[..., None]).squeeze(axis=-1)


def matVecInv(mat, u):
    r"""
    Compute vectorized Matrix Vector Inversion :math:`A^{-1}x` (A / x)

    Parameters
    ----------
    mat : np.ndarray, size (nDOF, M, M) or (M, M)
        Matrix or array of matrices.
    u : np.ndarray, size (nDOF, M) or (M,)
        Vector or array of vectors.

    Returns
    -------
    out : np.ndarray, size (nDOF, M) or (M,)
        The computed matrix-vector inversion(s)

    Notes
    -----
    - matVecInv((nDOF, M, M), (nDOF, M)) -> (nDOF, M) <=> (M, M) \ (M,) for each nDOF
    - matVecInv((M, M), (nDOF, M)) -> (nDOF, M) <=> (M, M) \ (M,) for each nDOF
    - matVecInv((M, M), (M,)) -> (M,) <=> (M, M) \ (M,)
    """
    try:
        return np.linalg.solve(mat, u)
    except ValueError:
        return np.linalg.solve(mat[None, ...], u)


def matMatMul(m1, m2):
    """
    Compute vectorized Matrix Matrix Multiplication :math:`AB` (A @ B)

    Parameters
    ----------
    m1 : np.ndarray, size (M, M)
        First matrix (single).
    m2 : np.ndarray, size (M, M, nDOF)
        Array of second matrices (multiple).

    Returns
    -------
    out : np.ndarray, size (M, M, nDOF)
        The computed matrix-matrix product.

    Notes
    -----
    - matMatMul((M, M), (M, M, nDOF)) -> (M, M, nDOF) <=> (M, M) @ (M, M) for each nDOF
    """
    return (m1 @ m2.transpose((1, 0, -1))).transpose((1,0,-1))
