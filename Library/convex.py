# -*- coding: utf-8 -*-
"""
Implementação de rotinas de estimação utilizando cvxpy

@author: Marcelo Daher
"""

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

SOLVER = "ECOS" # ECOS, ECOS_BB, SCS

args_lasso = {'solver': 'ECOS_BB',
           'max_iters': 2000}

args_bp = {'solver': 'SCS'}

args_bpdn = {'solver': 'SCS'}

args_tv = {'solver': 'SCS'}

def loss_fn(X, Y, beta):
    return cp.norm(cp.matmul(X, beta) - Y)**2


def regularizer(beta):
    return cp.norm1(beta)


def objective_fn(X, Y, beta, lambd):
    return loss_fn(X, Y, beta) + lambd * regularizer(beta)


def mse(X, Y, beta):
    return (1.0 / X.shape[0]) * loss_fn(X, Y, beta).value


def generate_data(m=100, n=20, sigma=5, density=0.2):
    "Generates data matrix X and observations Y."
    np.random.seed(1)
    beta_star = np.random.randn(n)
    idxs = np.random.choice(range(n), int((1-density)*n), replace=False)
    for idx in idxs:
        beta_star[idx] = 0
    X = np.random.randn(m,n)
    Y = X.dot(beta_star) + np.random.normal(0, sigma, size=m)
    return X, Y, beta_star


def plot_train_test_errors(train_errors, test_errors, lambd_values):
    plt.plot(lambd_values, train_errors, label="Train error")
    plt.plot(lambd_values, test_errors, label="Test error")
    plt.xscale("log")
    plt.legend(loc="upper left")
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.title("Mean Squared Error (MSE)")
    plt.show()


def plot_regularization_path(lambd_values, beta_values):
    num_coeffs = len(beta_values[0])
    for i in range(num_coeffs):
        plt.plot(lambd_values, [wi[i] for wi in beta_values])
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.xscale("log")
    plt.title("Regularization Path")
    plt.show()
    
def lasso(A,b,lambd,verbose = False):
    beta = cp.Variable(A.shape[1],complex=True)
    problem = cp.Problem(cp.Minimize(objective_fn(A, b, beta, lambd)))
    problem.solve(verbose = verbose, **args_lasso)
    return beta.value
    
def bp(A,b,verbose = False):
    '''
    solver for basis pursuit problem
    min|x|_{L1} for Ax = b
    '''
    x = cp.Variable(A.shape[1],complex=True)
    objective = cp.Minimize(cp.norm1(x))
    constraints = [A * x == b]
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose = verbose, **args_bp)
    return x.value

def bpdn(A,b,epsilon,verbose = False):
    '''
    solver for basis pursuit denoise problem
    min|x|_{L1} for ||b-Ax||**2 < epsilon
    '''
    x = cp.Variable(A.shape[1],complex=True)
    objective = cp.Minimize(cp.norm1(x))
    constraints = [loss_fn(A, b, x) <= epsilon]
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose = verbose, **args_bpdn)
    return x.value

def tv(Ax,Ay,B,verbose = False):
    '''
    minimizes total variation
    min|x|_{TV} for vec(Ax kron Ay * X) = b
    '''
    X = cp.Variable([Ay.shape[1],Ax.shape[1]],complex=True)
    objective = cp.Minimize(cp.tv(X))
    constraints = [Ay * X * Ax.T == B]
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose = verbose, **args_tv)
    return X.value


if __name__ == "__main__":
    
    vec = lambda x: np.reshape(x,np.product(x.shape))
    
    __import__("Export variables")
    
    print("Solving Lasso problem")
    y_lasso = lasso(V,x_vec,10**2,verbose = True)
    
    print("Solving Basis-Pursuit problem")
    y_bp = bp(V,x_vec,verbose = True)
    
    print("Solving BPDN problem")
    y_bpdn = bpdn(V,x_vec,10**-2,verbose = True)
    
    print("Solving TV problem")
    y_tv = tv(Vx,Vy,x,verbose = True)
    
    plt.figure(figsize=[12,4])
    plt.subplot(131)
    plt.imshow(np.abs(yFFS)**2)
    plt.gca().set_title("Referencia")
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(np.abs(np.reshape(y_lasso,fieldRes))**2)
    plt.gca().set_title("Lasso")
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(np.abs(np.reshape(vec(yFFS)-y_lasso,fieldRes)))
    plt.gca().set_title("Erro")
    plt.colorbar()
    
    plt.figure(figsize=[12,4])
    plt.subplot(131)
    plt.imshow(np.abs(yFFS)**2)
    plt.gca().set_title("Referencia")
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(np.abs(np.reshape(y_bp,fieldRes))**2)
    plt.gca().set_title("BP")
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(np.abs(np.reshape(vec(yFFS)-y_bp,fieldRes)))
    plt.gca().set_title("Erro")
    plt.colorbar()
    
    plt.figure(figsize=[12,4])
    plt.subplot(131)
    plt.imshow(np.abs(yFFS)**2)
    plt.gca().set_title("Referencia")
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(np.abs(np.reshape(y_bpdn,fieldRes))**2)
    plt.gca().set_title("BDNP")
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(np.abs(np.reshape(vec(yFFS)-y_bpdn,fieldRes)))
    plt.gca().set_title("Erro")
    plt.colorbar()
    
    plt.figure(figsize=[12,4])
    plt.subplot(131)
    plt.imshow(np.abs(yFFS)**2)
    plt.gca().set_title("Referencia")
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(np.abs(np.reshape(y_tv,fieldRes))**2)
    plt.gca().set_title("TV")
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(np.abs(np.reshape(vec(yFFS)-vec(y_tv),fieldRes)))
    plt.gca().set_title("Erro")
    plt.colorbar()
    
    
