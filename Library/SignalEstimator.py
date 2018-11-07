# coding=utf-8

import numpy as np

# para metodos regularizados
from numpy.linalg import norm
import scipy.sparse.linalg

from sys import version_info
if(version_info >= (3,0)):
    from importlib import reload as reload

import FarFieldSignal
reload(FarFieldSignal)
from FarFieldSignal import *

import GridArray
reload(GridArray)
from GridArray import *

import Imager
reload(Imager)
from Imager import *

import operations
reload(operations)
from operations import *

import omp
reload(omp)
from omp import *

from numpy import pi
from scipy.linalg import lstsq

class SignalEstimator(Imager):
    def __init__(self, gridArray, freq,
                 farFieldReference = None,
                 fieldRes = None):
        Imager.__init__(self, gridArray, freq,
                 farFieldReference = farFieldReference,
                 fieldRes = fieldRes)
        
    def estimator_mode(self, mode, **kwargs):
        if mode == "OMP":
            if kwargs['sparsity'] < 1:
                s = np.floor(kwargs['sparsity'] * self.nDir)
            else:
                s = int(kwargs['sparsity'])
            if 'eps' in kwargs:
                eps = kwargs['eps']
            else:
                eps = 0
            if 'x0' in kwargs:
                x0 = kwargs['x0']
            else:
                x0 = None
            estimator = lambda x: OMP(self.V, x, s, eps = eps, x0 = x0)[0]
            
        if mode == "OMP_kron":
            if kwargs['sparsity'] < 1:
                s = np.floor(kwargs['sparsity'] * self.nDir)
            else:
                s = int(kwargs['sparsity'])
            if 'eps' in kwargs:
                eps = kwargs['eps']
            else:
                eps = 0
            if 'x0' in kwargs:
                x0 = kwargs['x0']
            else:
                x0 = None
            estimator = lambda x: OMP_kron(self.Vx,self.Vy, x, s, eps = eps, x0 = x0)[0]
        
        if mode == "l0LS":
            if 'l0LS_params' in kwargs:
                l0LS_params = kwargs['l0LS_params']
            else:
                l0LS_params = [0.99,0.01,100]
            estimator = lambda x: l0LS(self.V,x,*l0LS_params)
        
        if mode == "least-squares":
            estimator = lambda x: np.linalg.lstsq(self.V, x)[0]
            
        # Convex estimators
        
        if mode in ["lasso","basis-pursuit","BPDN","TV"]:
            import convex
            reload(convex)
            
        if mode == "lasso":
            estimator = lambda x: convex.lasso(self.V, x, kwargs['lambd'])
            
        if mode == "basis-pursuit":
            estimator = lambda x: convex.bp(self.V, x)
            
        if mode == "BPDN":
            estimator = lambda x: convex.bpdn(self.V, x,kwargs['noise'])
            
        if mode == "TV":
            estimator = lambda x: np.reshape(convex.tv(self.Vx, self.Vy,
                                             np.reshape(x,[self.nMicX,self.nMicY])),
                                             self.nDir)
        
        # Beamformer estimators
        
        if mode in ["DAS", "Barlett", "MPDR", "MVDR"]:
            self.beamformer = mode
            estimator = lambda x: np.dot(self.wH, x)
            
        if mode == "DAS_kron":
            estimator = lambda x: self.Vy.T.conj().dot(np.reshape(x,[self.nMicX,self.nMicY])).dot(self.Vx.conj()).flatten()/self.nMic
        
        return estimator
        
        
    def plain_estimator(self, mode, **kwargs):
       
        nBins = np.shape(self.gridArray.signalFFT)[2]
        nFreq = np.shape(self.gridArray.signalFFT)[3]

        y = np.zeros([self.nDir, nFreq, nBins], dtype = np.complex128)
        
        estimator = self.estimator_mode(mode,**kwargs)
        
        freqs = np.fft.rfftfreq(nFreq)*self.gridArray.sampleRate
        freq0 = self.freq
        for freq in freqs:
            self.setFreq(freq)
            x = np.reshape(self.gridArray.signalFFT[:,:,:,self.freqBin],
                           [self.nMic, nBins])
            for bn in range(nBins):
                y[:,self.freqBin,bn] = estimator(x[:,bn])
        
        self.setFreq(freq0)
        return y
    
    def OMP_progressive(self, **kwargs):   
        # utilizando xhat da frequencia anterior como x0 da proxima
        
        nBins = np.shape(self.gridArray.signalFFT)[2]
        nFreq = np.shape(self.gridArray.signalFFT)[3]

        y = np.zeros([self.nDir, nFreq, nBins], dtype = np.complex128)
        freq0 = self.freq
        if kwargs['sparsity'] < 1:
            s = np.floor(kwargs['sparsity'] * self.nDir)
        else:
            s = int(kwargs['sparsity'])
        half_s = int(np.ceil(0.9*s))
        freqs = np.fft.rfftfreq(nFreq)*self.gridArray.sampleRate
        for freq in freqs:
            self.setFreq(freq)
            V = self.V
            x = np.reshape(self.gridArray.signalFFT[:,:,:,self.freqBin],
                           [self.nMic, nBins])
            xhat = None
            for bn in range(nBins):
                xhat, I = OMP(V, x, s, eps = 0,
                             x0 = xhat)
                y[:,self.freqBin,bn][I] = xhat[I]
                try:
                    xhat[I][np.argsort(np.abs(xhat[I]))][:half_s] = np.zeros(half_s)
                except:
                    shp = xhat[I][np.argsort(np.abs(xhat[I]))][:half_s].shape
                    xhat[I][np.argsort(np.abs(xhat[I]))][:half_s] = np.zeros(shp)
                
        
        self.setFreq(freq0)
        return y
    
        # utilizando xhat da frequencia anterior como x0 da proxima
    def l0LS_progressive(self,**kwargs):
        
        if 'l0LS_params' in kwargs:
            l0LS_params = kwargs['l0LS_params']
        else:
            l0LS_params = [0.99,0.01,100]
        
        nBins = np.shape(self.gridArray.signalFFT)[2]
        nFreq = np.shape(self.gridArray.signalFFT)[3]

        y = np.zeros([self.nDir, nFreq, nBins], dtype = np.complex128)
        
        freqs = np.fft.rfftfreq(nFreq)*self.gridArray.sampleRate
        for freq in freqs:
            self.setFreq(freq)
            V = self.V
            x = np.reshape(self.gridArray.signalFFT[:,:,:,self.freqBin],
                           [self.nMic, nBins])
            for bn in range(nBins):
                y[:,self.freqBin,bn] = l0LS(V,x[:,bn],*l0LS_params,
                                            x0 = y[:,self.freqBin-1,bn])
        return y
    
    
    def DAMAS2_LSQR(self, **kwargs):
        
        nBins = np.shape(self.gridArray.signalFFT)[2]
        nFreq = np.shape(self.gridArray.signalFFT)[3]

        y = np.zeros([self.nDir, nFreq, nBins], dtype = np.complex128)
        
        if kwargs['sparsity'] < 1:
            s = np.floor(kwargs['sparsity'] * self.nDir)
        else:
            s = int(kwargs['sparsity'])
        freqs = np.fft.rfftfreq(nFreq)*self.gridArray.sampleRate
        
        # average image to use same support
        avg_img = np.zeros([self.nDir])
        for freq in freqs:
            self.setFreq(freq)
            self.beamform(mode = "CSM-KAT", verbose = False)
            self.DAMAS2(1000,mode="KAT", verbose = False)
            
            avg_img += np.reshape(self.damas2Img,[self.nDir])
            
        support = np.reshape((avg_img != 0),[self.nDir])
        l0 = sum(support) # l0 norm of the proposed solution
        if l0 > s:
            # limita para os s maiores valores na imagem
            new_zeros = np.argsort(avg_img)[:self.nDir-s]
            for index in new_zeros:
                support[index] = False
            l0 = s
                
        for freq in freqs:
            self.setFreq(freq)
            
            V = self.V
            x = np.reshape(self.gridArray.signalFFT[:,:,:,self.freqBin],
                           [self.nMic, nBins])
            for bn in range(nBins):
                yhat = lstsq(V[:,support],x)[0]
                y[:,self.freqBin,bn][support] = yhat.reshape(l0)
        return y
    
    def beamform_estimator(self, beamformer = "DAS", verbose = True, **kwargs):
        # reconstruction from frequency domain
        
        self.beamformer = beamformer
        if verbose: print("Beamformer mode {}".format(beamformer))
        
        nBins = np.shape(self.gridArray.signalFFT)[2]
        
        if 'fieldRes' in kwargs:
            self.fieldRes = kwargs['fieldRes']
            self.nDirX = self.fieldRes[0]
            self.nDirY = self.fieldRes[1]
            self.nDir = np.product(self.fieldRes)
                    
        if self.beamformer in ["DAS", "Barlett", "MPDR"]:
            signalVector = np.reshape(
                    self.gridArray.signalFFT[:,:,:,self.freqBin],
                                [self.nMic,nBins])
            # matriz para conter a fft da estimativa
            y = np.zeros([self.nDir,nBins], dtype = np.complex128)
            wH = self.wH
            '''
            for b in range(nBins):
                y[:,b] = np.dot(wH,signalVector[:,b])
            '''
            y = np.apply_along_axis(lambda x: np.dot(wH,x),0,signalVector)
            Y = np.reshape(y,self.fieldRes)
        
        if self.beamformer == "X-KAT":
            # X.shape = [nMicX, nMicY, nBins]
            X = self.gridArray.signalFFT[:,:,:,self.freqBin]
            Y = np.zeros([self.nDirX, self.nDirY, nBins], dtype = np.complex128)
            Vxc = self.Vx.conj()
            VyH = self.Vy.T.conj()
            for i in range(nBins):
                Y[:,:,i] = VyH.dot(X[:,:,i]).dot(Vxc)

        #Y = Y/np.max(Y)
        self.bfImg = Y
        return np.copy(self.bfImg)
