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

import operations
reload(operations)
from operations import *

import omp
reload(omp)
from omp import *

import convex
reload(convex)

from numpy import pi
from scipy.linalg import lstsq

class Imager(object):
    def __init__(self, gridArray, freq,
                 farFieldReference = None,
                 fieldRes = None):
        self.gridArray = gridArray
        self.sampleRate = gridArray.sampleRate
        
        self.nMicX = gridArray.nMicX
        self.nMicY = gridArray.nMicY
        self.nMic = gridArray.nMic
        
        self.setFreq(freq)
        self.farFieldReference = farFieldReference
        if farFieldReference is not None:
            if fieldRes is None:
                self.fieldRes = farFieldReference.fieldRes
            else:
                self.fieldRes = fieldRes
            tao = gridArray.calculateDelays(farFieldReference)
        else:
            self.fieldRes = fieldRes
            reconstructedField = FarFieldSignal(fieldRes)
            tao = self.gridArray.calculateDelays(reconstructedField)
            del reconstructedField
        
        self.nDirX = self.fieldRes[0]
        self.nDirY = self.fieldRes[1]
        self.nDir = np.product(self.fieldRes)
        self.tao = np.reshape(tao,[self.nMic,self.nDir])
        
        self.beamformer = None
        self.DAMAS2mode = None
        self.damas2Iterations = 0
        
        self.psf = None
        self.A = None
        
        self.Rn_regularization = 0#1e-6
        self.Rx_regularization = 0#1e-6
        
    @property
    def Vx(self):
        microphonesX = self.gridArray.microphonesX
        dx = np.linspace(-1,1,self.fieldRes[0])
        '''
        Vx = np.zeros([self.nMicX,self.fieldRes[0]], dtype = 'complex128')
        # para cada microfone
        for x in range(self.nMicX):
            Vx[x] = dx*microphonesX[x]
        '''
        Vx = (np.reshape(microphonesX,[self.nMicY,1])).dot(np.reshape(dx,[self.fieldRes[0],1]).T)
        Vx = np.exp(-1j*Vx*2*pi*self.freq/self.gridArray.c)
        return Vx
    
    @property
    def Vy(self):
        microphonesY = self.gridArray.microphonesY
        dy = np.linspace(-1,1,self.fieldRes[1])
        '''
        Vy = np.zeros([self.nMicY,self.fieldRes[1]], dtype = 'complex128')
        # para cada microfone
        for y in range(self.nMicY):
            Vy[y] = dy*microphonesY[y]
        '''
        Vy = (np.reshape(microphonesY,[self.nMicY,1])).dot(np.reshape(dy,[self.fieldRes[1],1]).T)

        Vy = np.exp(-1j*Vy*2*pi*self.freq/self.gridArray.c)
        return Vy
        
    @property
    def V(self):
        #return np.power(np.e,-1j*self.tao*2*pi*self.freq)
        return np.kron(self.Vx,self.Vy)
    
    @property
    def w(self):
        if self.beamformer in ["DAS","X-KAT","CSM-KAT"]:
            return self.V/self.nMic
        if self.beamformer == "Barlett":
            return self.V/np.sqrt(self.nMic)
        if self.beamformer == "MPDR":
            V = self.V
            VHRinv = np.conj(V.T).dot(np.linalg.pinv(self.Rx+self.Rx_regularization*np.eye(self.nMic)))
            '''
            wH = np.zeros(V.shape, dtype = np.complex128).T
            for i in range(self.nDir):
                wH[i] = VHRinv[i]/np.dot(VHRinv[i],V[:,i])
            '''
            w_divisor = np.sum(VHRinv*V.T,axis=-1)
            wH = VHRinv/w_divisor[:,None]
            return np.conj(wH.T)
        if self.beamformer == "MVDR":
            V = self.V
            VHRinv = np.conj(V.T).dot(np.linalg.pinv(self.Rn+self.Rn_regularization*np.eye(self.nMic)))
            '''
            wH = np.zeros(V.shape, dtype = np.complex128).T
            for i in range(self.nDir):
                wH[i] = VHRinv[i]/np.dot(VHRinv[i],V[:,i])
            '''
            w_divisor = np.sum(VHRinv*V.T,axis=-1)
            WH = VHRinv/w_divisor[:,None]
            return np.conj(WH.T)
            
            
    @property
    def wH(self):
        return np.conj(self.w.T)
    
    #@property
    def calculateRn(self):
        nBins = np.shape(self.gridArray.noiseFFT)[2]
        signal = np.reshape(self.gridArray.noiseFFT[:,:,:,self.freqBin],
                                [self.nMic,nBins])
        if nBins == 1:
            self.Rn = signal.dot(signal.T.conj())
            return self.Rn#signal.dot(signal.T.conj())
        self.Rn = np.cov(signal,ddof=0)
        return self.Rn#np.cov(signal,ddof=0)
        
    @property
    def Rx(self):
        nBins = np.shape(self.gridArray.signalFFT)[2]
        signal = np.reshape(self.gridArray.signalFFT[:,:,:,self.freqBin],
                                [self.nMic,nBins])
        if nBins == 1:
            return signal.dot(signal.T.conj())
        return np.cov(signal,ddof=0)
    
    def setFreq(self, freq):
        windowFFT = int(self.gridArray.windowFFT)
        
        #frequencias em rad/s
        freqs = np.fft.rfftfreq(windowFFT)*self.gridArray.sampleRate
        
        #nao comecar do zero para nao pegar nivel DC para frequencias baixas
        #f = 1
        #while np.abs(freq-freqs[f]) > np.abs(freq-freqs[f+1]):
        #    f = f+1
        f = np.argmin(np.abs(freqs-freq))
        self.freq = freqs[f]
        self.freqBin = f
        return self.freq
        
    def calculateA(self):
        '''
        V = self.V
        A = np.zeros([self.nMic**2, self.nDir], dtype = np.complex128)
        for i in range(self.nDir):
            v = V[:,i]
            A[:,i] = np.kron(v.conj(),v)
        '''
        V = self.V
        A = colKRproduct_conj_self(V)
        self.A = np.nan_to_num(A)
    
    def exportVs(self,**kwargs):
        '''
        Exports parameters Vx, Vy variables to CSV file
        '''
        #if 'delimiter' not in kwargs:
        #    kwargs['delimiter'] = ';'
        
        kwargs['fname'] = 'Vx_real.csv'
        kwargs['X'] = self.Vx.real
        np.savetxt(**kwargs)
        
        kwargs['fname'] = 'Vx_imag.csv'
        kwargs['X'] = self.Vx.imag
        np.savetxt(**kwargs)
        
        kwargs['fname'] = 'Vy_real.csv'
        kwargs['X'] = self.Vy.real
        np.savetxt(**kwargs)
        
        kwargs['fname'] = 'Vy_imag.csv'
        kwargs['X'] = self.Vy.imag
        np.savetxt(**kwargs)
        
    def beamform(self, beamformer = "DAS", verbose = True, **kwargs):
        # reconstruction from frequency domain
        
        self.beamformer = beamformer
        if verbose: print("Beamformer mode {}".format(beamformer))
        
        nBins = np.shape(self.gridArray.signalFFT)[2]
        
        if 'fieldRes' in kwargs:
            self.fieldRes = kwargs['fieldRes']
            self.nDirX = self.fieldRes[0]
            self.nDirY = self.fieldRes[1]
            self.nDir = np.product(self.fieldRes)
                    
        if self.beamformer in ["DAS", "Barlett", "MPDR", "MVDR"]:
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
            y = np.abs(y)**2
            y = np.mean(y,axis=-1)
            Y = np.reshape(y,self.fieldRes)
        
        if self.beamformer == "X-KAT":
            # X.shape = [nMicX, nMicY, nBins]
            X = self.gridArray.signalFFT[:,:,:,self.freqBin]
            Y = np.zeros([self.nDirX, self.nDirY, nBins], dtype = np.complex128)
            Vxc = self.Vx.conj()
            VyH = self.Vy.T.conj()
            for i in range(nBins):
                Y[:,:,i] = VyH.dot(X[:,:,i]).dot(Vxc)
            Y = np.abs(Y)**2
            Y = np.mean(Y,axis=-1)
            
        if self.beamformer == "CSM-KAT":
            Vx2 = colKRproduct_conj_self(self.Vx)
            Vy2 = colKRproduct_conj_self(self.Vy)
            
            Vy2H = Vy2.T.conj()
            Vx2c = Vx2.conj()
            
            Z = S2Z(self.Rx,self.nMicX,self.nMicY)
            
            Y = Vy2H.dot(Z).dot(Vx2c)
            
            Y = np.transpose(np.abs(Y))

        #Y = Y/np.max(Y)
        self.bfImg = Y
        return np.copy(self.bfImg)

######################################################
# Deconvolution algorithms
#
#
#
#

    def DAMAS2(self, iterations, verbose = True, **kwargs):
        
        iterations = int(iterations)
        if self.DAMAS2mode is None:
            if "mode" in kwargs:
                self.DAMAS2mode = kwargs["mode"]
            else:
                self.DAMAS2mode = "PSF"
                
        if verbose: print("DAMAS2 mode {}".format(self.DAMAS2mode))
        
        if "freq" in kwargs:
            self.freq = kwargs["freq"]
        if "mode" in kwargs:
            self.DAMAS2mode = kwargs["mode"]
        else:
            self.DAMAS2mode = "PSF"
        
        fieldRes = np.shape(self.bfImg)
        
        if self.damas2Iterations == 0:
            if self.beamformer == None:
                self.beamform(fieldRes)
                
            if self.DAMAS2mode == "PSF":
                if self.nDir%2:
                    center = int(self.nDir/2)
                else:
                    center = int(self.nDir/2 + fieldRes[1]/2)
                        
                psf = np.zeros(self.nDir,dtype = np.complex128)
        
                #calculando a PSF para a frequencia especificada
                v = self.V[:,center]
                #v = np.ones(self.nMic)
                w = self.w
                '''
                for u in range(self.nDir):
                    psf[u] = np.abs(np.conj(w[:,u]).dot(v))**2#*vH.dot(w[:,u])
                '''
                func = lambda x: np.abs(np.conj(x).dot(v))**2
                psf = np.apply_along_axis(func,0,w)
                
                self.psf = np.nan_to_num(psf)
            # end "PSF" if
            if self.DAMAS2mode == "A-matrix":
                self.calculateA()
            self.damas2Img = np.zeros(fieldRes)
        
        y0 = np.copy(self.bfImg)
        y = np.copy(self.damas2Img)
        
        if self.DAMAS2mode == "PSF":
            psf = np.copy(self.psf)
            psf = np.reshape(psf,fieldRes)
            a = np.nansum(psf)
            y0 = np.reshape(y0,fieldRes)
            y0 = np.nan_to_num(y0)
            
            from scipy.signal import fftconvolve
            
            for i in range(iterations):
                b = fftconvolve(y, psf, mode='same')
                y = np.maximum(y + (y0 - b)/a, 0)
        
        if self.DAMAS2mode == "A-matrix":
            y0 = np.reshape(y0,[self.nDir,1])
            y = np.reshape(y,[self.nDir,1])
            
            A = self.A
            AHA = np.conj(A.T).dot(A)
            a = np.nansum(np.abs(AHA))/(self.nDir)
            
            for i in range(iterations):
                b = AHA.dot(y)
                y = np.maximum(y + (y0 - b)/a, 0)
        
        if self.DAMAS2mode == "KAT":
            y0 = np.reshape(y0,fieldRes)
            
            Vy = self.Vy
            Vx = self.Vx
            VyHVy = np.abs(np.conj(Vy.T).dot(Vy))
            VxtVxc= np.abs(Vx.T.dot(np.conj(Vx)))
            
            a = np.sum(np.abs(np.kron(VxtVxc.conj(),VyHVy)))/(self.nDir)
            
            #tese flavio, pag 201 eq 40
            for i in range(iterations):
                b = VyHVy.dot(y).dot(VxtVxc)
                y = np.maximum(y + (y0 - b)/a, 0)
        
        self.damas2Iterations += iterations
        
        self.damas2Img = np.reshape(np.abs(y),self.fieldRes)
        return np.copy(y)


    def CLEAN(self, iterations = None, gama = None, **kwargs):
        return

    def beta_lsqr(self):
        
        x = np.reshape(self.gridArray.signalFFT[:,:,:,self.freqBin],self.nMic)
        V = self.V
        y0 = np.zeros(self.nDir,dtype=np.complex128)
        
        func = lambda y: norm(x-V.dot(y),ord=2)
        
        #return np.linalg.lstsq(V, x)
        return scipy.sparse.linalg.lsmr(V, x)
        
    def image_estimator(self, mode,returnSignal = False, **kwargs):
        
        nBins = np.shape(self.gridArray.signalFFT)[2]

        x = np.reshape(self.gridArray.signalFFT[:,:,:,self.freqBin],
                        [self.nMic, nBins])
        y = np.zeros([self.nDir, nBins], dtype = np.complex128)
        V = self.V
        
        if mode == "OMP":
            if kwargs['sparsity'] > 1:
                s = kwargs['sparsity']
            else:
                s = np.floor(kwargs['sparsity'] * self.nDir)
            if 'x0' in kwargs:
                x0 = kwargs['x0']
            else:
                x0 = None
            estimator = lambda x: OMP(V, x, s, eps = 0, x0 = x0)[0]
        
        if mode == "OMP2":
            from sklearn import linear_model
            if kwargs['sparsity'] > 1:
                s = kwargs['sparsity']
            else:
                s = np.floor(kwargs['sparsity'] * self.nDir)
            estimator = lambda x: linear_model.orthogonal_mp(
                                    V,x,n_nonzero_coefs=s)
        
        if mode == "l0LS":
            if 'l0LS_params' in kwargs:
                l0LS_params = kwargs['l0LS_params']
            else:
                l0LS_params = [0.99,0.01,100]
            estimator = lambda x: l0LS(V,x,*l0LS_params)
        
        if mode == "l0LS_kron":
            if 'l0LS_params' in kwargs:
                l0LS_params = kwargs['l0LS_params']
            else:
                l0LS_params = [0.99,0.01,100]
            Vx = self.Vx
            Vy = self.Vy
            estimator = lambda x: l0LS_kron(Vx,Vy,x,*l0LS_params)
            
        if mode == "least-squares":
            estimator = lambda x: np.linalg.lstsq(V, x)[0]
            
        if mode == "lasso":
            estimator = lambda x: convex.lasso(V, x, kwargs['lambd'])
            
        if mode == "basis-pursuit":
            estimator = lambda x: convex.bp(V, x)
            
        if mode == "BPDN":
            estimator = lambda x: convex.bpdn(V, x,kwargs['noise'])
            
        if mode == "TV":
            estimator = lambda x: np.reshape(convex.tv(self.Vx, self.Vy,
                                             np.reshape(x,[self.nMicX,self.nMicY])),
                                             self.nDir)
            
        for i in range(nBins):
            y[:,i] = estimator(x[:,i])
        
        if returnSignal: return y
        Y = np.reshape(np.mean(np.abs(y)**2,axis=-1),self.fieldRes)
        return Y
    
    def signal_estimator(self, mode, **kwargs):
       
        nBins = np.shape(self.gridArray.signalFFT)[2]
        nFreq = np.shape(self.gridArray.signalFFT)[3]

        y = np.zeros([self.nDir, nFreq, nBins], dtype = np.complex128)
        
        if mode == "OMP":
#            s = np.floor(sparsity * self.nDir)
#            estimator = lambda x: OMP(V, x, s, eps = 0, x0 = None)[0]
#            pass
            
            freq0 = self.freq
            
            if kwargs['sparsity'] < 1:
                s = np.floor(kwargs['sparsity'] * self.nDir)
            else:
                s = int(kwargs['sparsity'])
            freqs = np.fft.rfftfreq(nFreq)*self.gridArray.sampleRate
            for freq in freqs:
                self.setFreq(freq)
                V = self.V
                x = np.reshape(self.gridArray.signalFFT[:,:,:,self.freqBin],
                               [self.nMic, nBins])
                for bn in range(nBins):
                    xhat, I = OMP(V, x, s)
                    y[:,self.freqBin,bn][I] = xhat[I]
            
            self.setFreq(freq0)
            return y
        
        if mode == "OMP2":
            if kwargs['sparsity'] < 1:
                s = np.floor(kwargs['sparsity'] * self.nDir)
            else:
                s = int(kwargs['sparsity'])
            estimator = lambda x: linear_model.orthogonal_mp(
                                    V,x,n_nonzero_coefs=s)
        
        if mode == "l0LS":
            if 'l0LS_params' in kwargs:
                l0LS_params = kwargs['l0LS_params']
            else:
                l0LS_params = [0.99,0.01,100]
            estimator = lambda x: l0LS(V,x,*l0LS_params)
        
        if mode == "least-squares":
            estimator = lambda x: np.linalg.lstsq(V, x)[0]
            
        # utilizando xhat da frequencia anterior como x0 da proxima
        if mode == "l0LS-progressive":
            freqs = np.fft.rfftfreq(nFreq)*self.gridArray.sampleRate
            for freq in freqs:
                self.setFreq(freq)
                V = self.V
                x = np.reshape(self.gridArray.signalFFT[:,:,:,self.freqBin],
                               [self.nMic, nBins])
                for bn in range(nBins):
                    y[:,self.freqBin,bn] = l0LS(V,x[:,bn],0.9,0.1,100,
                                                x0 = y[:,self.freqBin-1,bn])
            return y
        
        if mode == "least-squares":
            estimator = lambda x: np.linalg.lstsq(V, x)[0]
            
        if mode == "lasso":
            estimator = lambda x: convex.lasso(V, x, kwargs['lambd'])
            
        if mode == "basis-pursuit":
            estimator = lambda x: convex.bp(V, x)
            
        if mode == "BPDN":
            estimator = lambda x: convex.bpdn(V, x,kwargs['noise'])
            
        if mode == "TV":
            estimator = lambda x: np.reshape(convex.tv(self.Vx, self.Vy,
                                             np.reshape(x,[self.nMicX,self.nMicY])),
                                             self.nDir)
            
        # utilizando xhat da frequencia anterior como x0 da proxima
        if mode == "OMP-progressive":
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
        
        if mode == "DAMAS2-LSQR":
            # Estima o suporte pelo DAMAS2
            # Estima o sinal por MMQ
            if kwargs['sparsity'] < 1:
                s = np.floor(kwargs['sparsity'] * self.nDir)
            else:
                s = int(kwargs['sparsity'])
            half_s = int(np.ceil(0.9*s))
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
                xhat = None
                for bn in range(nBins):
                    yhat = lstsq(V[:,support],x)[0]
                    y[:,self.freqBin,bn][support] = yhat.reshape(l0)
            return y
        
        freqs = np.fft.rfftfreq(nFreq)*self.gridArray.sampleRate
        for freq in freqs:
            self.setFreq(freq)
            V = self.V
            x = np.reshape(self.gridArray.signalFFT[:,:,:,self.freqBin],
                           [self.nMic, nBins])
            for bn in range(nBins):
                y[:,self.freqBin,bn] = estimator(x[:,bn])
        
        return y