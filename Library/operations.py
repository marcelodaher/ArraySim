# coding=utf-8

import numpy as np

def colKRproduct(A,B):
    '''
    columnwise Khatri-Rao product between matrix A and B
    '''
    if A.shape[1] != B.shape[1]:
        raise TypeError("A and B must have the same number of columns")
    q = A.shape[1]
    C = np.zeros([A.shape[0]*B.shape[0],q])
    for i in np.arange(q):
        C[:,i] = np.kron(A[:,i],B[:,i])
    return C

def colKRproduct_conj_self(A):
    return np.apply_along_axis(lambda x: np.kron(x.conj(),x),0,A)

def Xi(nMicX,nMicY):
    '''
    Retorna a matrix de permutação \Xi
    '''
    Xi = np.zeros([nMicX*nMicY,nMicX*nMicY])
    
    print("XI() NOT IMPLEMENTED")
    
    return Xi

def S2Z(S,nMicX,nMicY):
    
    Z = np.zeros([nMicX*nMicY,nMicX*nMicY], dtype = S.dtype)
    
    for x in np.arange(nMicX):
        for y in np.arange(nMicX):
            Z[:,y+x*nMicY] = np.reshape(
                                S[y*nMicY:(y+1)*nMicY,x*nMicX:(x+1)*nMicX],
                                newshape = [nMicX*nMicY],
                                order="F")
    return Z
    
def spark(A):
    from itertools import combinations as comb
    from  numpy import linalg
    A = np.array(A)
    At = A.T
    [m,n] = At.shape
    if n > m: return 0
    for k in range (1,n+1):
        row_combos = comb(range(m),k)
        for rows in row_combos:
            R = np.array([At[row] for row in rows])
            rank = linalg.matrix_rank(R)
            if rank < k: return k
    return n+1
    