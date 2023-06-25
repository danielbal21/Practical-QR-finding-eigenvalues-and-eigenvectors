import sys

import numpy as np
from time import time

eps = 1.0e-5

def fast_norm(A):
    return np.sqrt(np.sum(np.square(A)))

def wilkinson_shift(T):
    mu = 0
    m = T.shape[0]
    if m == 1:
        mu = T[0, 0]
    else:
        B = T[m - 2:, m - 2:]  # rightmost 2x2
        delta = (B[0, 0] - B[1, 1]) / 2
        if delta == 0:  # Arbitrary set sign if delta ---> 0
            sign = 1
        else:
            sign = np.sign(delta)
        # Calculate mu
        mu = B[1, 1] - sign * B[0, 1] * B[1, 0] / (abs(delta) + np.sqrt(delta ** 2 + B[0, 1] * B[1, 0]))
        #mu = B[1, 1] + delta - sign * np.sqrt(delta ** 2 + (B[1, 0] ** 2))
        #mu = sign * B[1, 0] * B[1, 0] / (abs(delta) + np.sqrt((delta * delta) + (B[0, 0] * B[0, 0])))
    return mu

def householder_fast_QR(A):
    n = A.shape[0]
    Q = np.asanyarray(np.eye(n), dtype=np.float32)
    R = np.asanyarray(A.copy(), dtype=np.float32)
    for j in range(n - 1):
        vj = np.asanyarray(R[j:, j].copy(),dtype=np.float32)
        vj[0] += np.sign(vj[0]) * fast_norm(vj)
        vj /= fast_norm(vj)

        # Apply the Householder reflector to R
        R[j:, j:] -= 2 * np.outer(vj, np.dot(vj, R[j:, j:]))
        # Apply the Householder reflector to Q
        Q[j:] -= 2 * np.outer(vj, np.dot(vj, Q[j:]))
    return Q.T, R

def householder_fast_QR_test(A):
    n = A.shape[0]
    Q = np.asanyarray(np.eye(n), dtype=np.float32)
    R = np.asanyarray(A.copy(), dtype=np.float32)
    for j in range(n - 1):
        vj = np.asanyarray(R[j:j+2, j].copy(),dtype=np.float32)
        vj[0] += np.sign(vj[0]) * fast_norm(vj)
        vj /= fast_norm(vj)
        # Apply the Householder reflector to R
        R[j:j+2, j:] -= 2 * np.outer(vj, np.dot(vj, R[j:j+2, j:]))
        # Apply the Householder reflector to Q
        Q[j:j+2] -= 2 * np.outer(vj, np.dot(vj, Q[j:j+2]))
    return Q.T, R


def SQR(A, eps, d_type=np.float32):
    n = A.shape[0]
    B, P = np.asanyarray(A.copy(), dtype=d_type), np.eye(n, dtype=d_type)
    I = np.eye(n, dtype=d_type)
    mu = 0
    is_null = True

    if n == 1:
        return np.diag(A), np.eye(1, dtype=d_type)

    while is_null:
        Q, R = householder_fast_QR_test(B - mu * I)
        B = R @ Q + mu * I
        P = P @ Q
        mu = wilkinson_shift(B)
        d = np.abs(np.diag(B, -1))
        off_diag_min = d.min()
        off_diag_min_ind = d.argmin() + 1
        is_null = off_diag_min > eps

    t = off_diag_min_ind

    eval1, evec1 = SQR(B[:t, :t], eps)
    eval2, evec2 = SQR(B[t:, t:], eps)

    v1 = np.r_[evec1, np.zeros([evec2.shape[0], evec1.shape[1]])]
    v2 = np.r_[np.zeros([evec1.shape[0], evec2.shape[1]]), evec2]

    return np.c_[eval1, eval2], P @ np.c_[v1, v2]

def hessenberg(matrix):
    n = matrix.shape[0]
    hessenberg_matrix = np.copy(matrix)
    transformation_matrices = []

    for k in range(1, n - 1):
        x = hessenberg_matrix[k:, k - 1]
        v = np.zeros_like(x)
        v[0] = np.sign(x[0]) * fast_norm(x)

        if fast_norm(x) != 0:
            u = x + v
            u /= np.linalg.norm(u)

            hessenberg_matrix[k:, k - 1:] -= 2.0 * np.outer(u, np.dot(u, hessenberg_matrix[k:, k - 1:]))

            hessenberg_matrix[:, k:] -= 2.0 * np.outer(np.dot(hessenberg_matrix[:, k:], u), u)

            transformation_matrix = np.identity(n)
            transformation_matrix[k:, k:] -= 2.0 * np.outer(u, u)
            transformation_matrices.append(transformation_matrix)

    return hessenberg_matrix, np.prod(transformation_matrices[::-1], axis=0).T

def test_eig(A, L, V):
    return np.max(np.linalg.norm(np.asanyarray(A @ V - V * L.reshape(1, -1), dtype=np.float32))) / np.linalg.norm(V)


def Read_Data(path, delim):
    data_matrix = np.loadtxt(path, dtype='f', delimiter=delim)
    return data_matrix


np.set_printoptions(precision=5, suppress=True)



file = 'inv_matrix(800 x 800).txt'
A = Read_Data(file, ' ')

# m = 8  # size of matrix
# A = np.random.rand(m, m)
# A = 0.5 * (A + A.T)


HA, Tr = hessenberg(A)


# Our calculations
tt = time()
print (A)
L_1, Vec_1 = SQR(HA, eps)
Vec_1 = Vec_1 @ Tr
Vec_1 = Vec_1 / np.sqrt(np.sum(Vec_1 ** 2, axis=0))
print("Error of our calculations (SQR)", test_eig(HA, L_1, Vec_1))  # Error of calculation ||HA*v - l*v||
print("Time of our calculations (SQR)", time() - tt)

tt = time()
L_2, Vec_2 = np.linalg.eig(A)
print("Error of (Linalg)", test_eig(HA, L_2, Vec_2))  # Error of calculation ||HA*v - l*v||
print("Time of (Linalg)", time() - tt)

# Sorting of eigenvalues and eigenvectors
ind_val_1 = np.argsort(L_1[0])  # indsort of SQR
ind_val_2 = np.argsort(L_2)  # indsort of Linalg

L_1 = L_1[:, ind_val_1]
Vec_1 = Vec_1[:, ind_val_1]

L_2 = L_2[ind_val_2]
Vec_2 = Vec_2[:, ind_val_2]

#set all below eps to zero
Vec_1[np.abs(Vec_1) < eps] = 0
Vec_2[np.abs(Vec_2) < eps] = 0

print("Our:")
print("eigenvalues: \n",L_1)
print("eigenvectors: \n",Vec_1)
print("---------------------------------------------------------")
print("Theirs:")
print("eigenvalues: \n",L_2)
print("eigenvectors: \n",Vec_2)