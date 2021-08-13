# Copyright (c) 2021 Jason Pereira <jason.pereira@york.ac.uk>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import random


def omega(d):
    omega_single = [[0, 1], [-1, 0]]
    return np.kron(np.identity(d), omega_single)


def denoms(symp_eigs, d):
    denom_list = []
    for m in range(d):
        prod = 1
        for n in range(d):
            if m != n:
                prod = prod * (symp_eigs[m] ** 2 - symp_eigs[n] ** 2)
        denom_list.append(symp_eigs[m] * prod)
    return denom_list


def vals(mat, signs, symp_eigs, denom_list, d):
    val_list = []
    for m in range(d):
        red_mat = np.delete(mat - symp_eigs[m] * 1j * omega(d), 2 * m + 1, 0)
        val_m_list = []
        for n in range(2 * d):
            val_m_list.append(np.linalg.det(np.delete(red_mat, n, 1)) / denom_list[m])
        val_list.append(val_m_list)
    for m in range(d):
        denom = np.sqrt(val_list[m][2 * m + 1])
        val_list[m] = list(map(lambda x: x / denom, np.multiply(val_list[m], signs)))
    return val_list


def symp_from_vals(val_list, d):
    S = np.zeros((2 * d, 2 * d))
    for m in range(d):
        for n in range(d):
            S[2 * m:2 * m + 2, 2 * n:2 * n + 2] = np.array([[np.real(val_list[m, 2 * n + 1]),
                                                             -np.real(val_list[m, 2 * n])],
                                                            [-np.imag(val_list[m, 2 * n + 1]),
                                                             np.imag(val_list[m, 2 * n])]])
    return S


def symp(mat):
    d = len(mat) // 2
    signs = list(map(lambda x: (-1) ** x, range(1, 2 * d + 1)))
    symp_eigs = np.real(np.linalg.eigvals(1j * np.dot(omega(d), mat)))
    symp_eigs = sorted(symp_eigs)[len(symp_eigs) // 2:]
    denom_list = denoms(symp_eigs, d)
    val_list = np.array(vals(mat, signs, symp_eigs, denom_list, d))
    S = symp_from_vals(val_list, d)
    return S


def chop(mat, eps=0.000001):
    new_mat = mat
    new_mat[np.abs(mat) < eps] = 0
    return new_mat


def symp_test(mat, eps=0.000001):
    d = len(mat) // 2
    test = np.dot(np.dot(mat, omega(d)), np.transpose(mat)) - omega(d)
    return np.all(np.abs(test) < eps)


d = 3
V = 2 * (np.random.rand(2 * d, 2 * d) - 1 / 2)
V = np.dot(np.transpose(V), V)
S = symp(V)

print("Covariance matrix")
print(V)
print("Verify calculated matrix is symplectic")
print(symp_test(S))
print("Diagonalise the covariance matrix")
print(chop(-np.dot(np.dot(np.dot(np.dot(S, omega(3)), V), omega(3)), np.transpose(S))))
