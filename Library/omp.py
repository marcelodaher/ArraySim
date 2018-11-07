# coding=utf-8

import numpy as np
import scipy

from scipy.linalg import lstsq

def OMP(A, y, s, eps = 0, x0 = None):
    
    # Solves Ax = y for x
    
    s = int(s)
    N = A.shape[1]
    
    y_cp = np.copy(y)
    y_cp = y_cp.flatten()
    r = np.copy(y_cp)
    
    if x0 is None:
        x0 = np.zeros(N, dtype = np.complex128)
        x = np.copy(x0)
        lambd = (x != 0)
    
    else:
        lambd = (x0 != 0)
        x = x0.astype(np.complex128)
        x[lambd] = lstsq(A[:,lambd],y)[0]
        
    x = np.copy(x0)
    t0 = sum(lambd) + 1
    
    for t in range(t0,s+1):
        ind = np.argmax(np.abs(r.dot(A.conj())))
        lambd[ind] = True
        
        #x[lambd] = np.linalg.lstsq(A[:,lambd],y_cp,rcond=None)[0]
        x[lambd] = lstsq(A[:,lambd],y_cp)[0]
        
        r = y_cp - A[:,lambd].dot(x[lambd])
        
        if (np.linalg.norm(r)/np.linalg.norm(y_cp)) < eps:
            return np.reshape(x,N), lambd
    return np.reshape(x,N), lambd

def OMP_kron(Ax,Ay, y, s, eps = 0, x0 = None):
    
    # Solves Ax = y for x
    
    s = int(s)
    N = Ax.shape[1]*Ay.shape[1]
    A = np.kron(Ax,Ay)
    Ayc = Ay.conj()
    AxH = Ax.T.conj()
    Axc = Ax.conj()
    AyH = Ay.T.conj()
    Rshape = [Ay.shape[0],Ax.shape[0]]
    r = y.flatten()
    
    if x0 is None:
        x0 = np.zeros(N, dtype = np.complex128)
        x = np.copy(x0)
        lambd = (x != 0)
    
    else:
        lambd = (x0 != 0)
        x = x0.astype(np.complex128)
        x[lambd] = lstsq(A[:,lambd],y)[0]
        
    t0 = sum(lambd) + 1
    
    for t in range(t0,s+1):
        #ind = np.argmax(np.abs(r.dot(A.conj())))
        ind = np.argmax(np.abs(np.dot(AxH,np.dot(np.reshape(r,Rshape),Ayc))))
        #ind = np.argmax(np.abs(np.dot(AyH,np.dot(np.reshape(r,Rshape),Axc))))
        lambd[ind] = True
        
        #x[lambd] = np.linalg.lstsq(A[:,lambd],y_cp,rcond=None)[0]
        x[lambd] = lstsq(A[:,lambd],y)[0]
        
        r = y - A[:,lambd].dot(x[lambd])
        
        if eps != 0:
            if (np.linalg.norm(r)/np.linalg.norm(y)) < eps:
                return np.reshape(x,N), lambd
    return np.reshape(x,N), lambd

def l0LS(A, y, gamma, mu, maxiter, x0 = None):
    b = np.dot(A.T.conj(),y)
    c = np.copy(b)
    R = np.dot(A.T.conj(),A)
    Rd = np.diag(R)
    lbd = np.max(0.5*np.abs(b)**2/Rd)
    lbdmin = mu*lbd
    
    if x0 is None:
        t = np.argmax(0.5*np.abs(b)**2/Rd)
        I = np.zeros((np.shape(A)[1]), dtype = np.bool)
        I[t] = True
        
        x = np.zeros(np.shape(A)[1], dtype = b.dtype)
        updated = False
        
    else:
        x = x0.astype(b.dtype)
        I = (x != 0)
        updated = True
    
    iter = 0
    while (lbd >= lbdmin and iter < maxiter):
        if updated:
            x[I] = lstsq(A[:,I],y)[0]
            c = b - np.dot(R,x)
        lbd = gamma * lbd
        updated = False
        
        
        remFun = 0.5*(np.abs(x)**2)*Rd + np.real(x.conj()*c)
        remFun[I == False] = np.inf
        remInd = np.argmin(remFun)
        remMin = remFun[remInd]
        while remMin < lbd:
            updated = True
            I[remInd] = False
            c = c + x[remInd]*R[:,remInd]
            x[remInd] = 0
            
            remFun = (0.5*np.abs(x)**2)*Rd + np.real(x.conj()*c)
            remFun[I == False] = np.inf
            remInd = np.argmin(remFun)
            remMin = remFun[remInd]
        addFun = 0.5*(np.abs(c)**2)/Rd
        addFun[I==True] = -1
        addInd = np.argmax(addFun)
        addMax = addFun[addInd]
        while addMax > lbd:
            updated = True
            I[addInd] = True
            x[addInd] = c[addInd]/Rd[addInd]
            c = c - x[addInd]*R[:,addInd]
            
            addFun = 0.5*(np.abs(c)**2)/Rd
            addFun[I==True] = -1
            addInd = np.argmax(addFun)
            addMax = addFun[addInd]
        iter = iter + 1
    if updated:
        x[I] = lstsq(A[:,I],y)[0]
    
    return x
    
def l0LS_kron(Ax,Ay, y, gamma, mu, maxiter, x0 = None):
    A = np.kron(Ax,Ay)
    
    Xshape = [Ay.shape[1],Ax.shape[1]]
    xshape = np.shape(A)[1]
    
    b = np.dot(A.T.conj(),y)
    bshape = b.shape
    c = np.copy(b)
    Rx = np.dot(Ax.T.conj(),Ax)
    Ry = np.dot(Ay.T.conj(),Ay)
    R = np.dot(A.T.conj(),A)
    Rd = np.diag(R)
    lbd = np.max(0.5*np.abs(b)**2/Rd)
    lbdmin = mu*lbd
    
    if x0 is None:
        t = np.argmax(0.5*np.abs(b)**2/Rd)
        I = np.zeros((np.shape(A)[1]), dtype = np.bool)
        I[t] = True
        
        x = np.zeros(xshape, dtype = b.dtype)
        updated = False
        
    else:
        x = x0.astype(b.dtype)
        I = (x != 0)
        updated = True
    
    iter = 0
    while (lbd >= lbdmin and iter < maxiter):
        if updated:
            x[I] = lstsq(A[:,I],y)[0]
            X = np.reshape(x,Xshape)
            c = b - np.reshape(Ry.dot(X).dot(Rx.T),bshape)
            #c = b - np.dot(R,x)
        lbd = gamma * lbd
        updated = False
        
        
        remFun = 0.5*(np.abs(x)**2)*Rd + np.real(x.conj()*c)
        remFun[I == False] = np.inf
        remInd = np.argmin(remFun)
        remMin = remFun[remInd]
        while remMin < lbd:
            updated = True
            I[remInd] = False
            c = c + x[remInd]*R[:,remInd]
            x[remInd] = 0
            
            remFun = (0.5*np.abs(x)**2)*Rd + np.real(x.conj()*c)
            remFun[I == False] = np.inf
            remInd = np.argmin(remFun)
            remMin = remFun[remInd]
        addFun = 0.5*(np.abs(c)**2)/Rd
        addFun[I==True] = -1
        addInd = np.argmax(addFun)
        addMax = addFun[addInd]
        while addMax > lbd:
            updated = True
            I[addInd] = True
            x[addInd] = c[addInd]/Rd[addInd]
            c = c - x[addInd]*R[:,addInd]
            
            addFun = 0.5*(np.abs(c)**2)/Rd
            addFun[I==True] = -1
            addInd = np.argmax(addFun)
            addMax = addFun[addInd]
        iter = iter + 1
    if updated:
        x[I] = lstsq(A[:,I],y)[0]
    
    return x
    