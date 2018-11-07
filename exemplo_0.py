# -*- coding: utf-8 -*-
"""
Example of usage for Acoustic Imaging Simulator
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time

import FarFieldSignal
reload(FarFieldSignal)
from FarFieldSignal import *

import GridArray
reload(GridArray)
from GridArray import *

import Imager
reload(Imager)
from Imager import Imager

# creating the signal
signal_N = 44100
fieldRes = np.array([32,32])

f = FarFieldSignal(fieldRes, sampleRate = 44100)

pos_sinal = np.round(fieldRes*np.array([2./4,2./4])).astype(np.int)
f.addSource(np.sin(2*np.arange(signal_N)),point=pos_sinal)
pos_ruido = np.round(fieldRes*np.array([1./3,1./3])).astype(np.int)
f.addNoise(variance = 1, bandwidth = 15e3,
           mode = 'point', point = pos_ruido)

# 8x8 minimum missing lags array
xs = np.array([0,1,4,9,15,22,32,34])*0.3/34
ys = xs
a = GridArray(xs, ys, sampleRate = 44100, verbose = True)

#a.show() # plots array geometry
a.receiveSignal(f)
a.addNoise(snr = 30) # 30 dB sampling noise

windowFFT = 44100
a.signal2FFT(windowFFT = windowFFT)

lookingFreq = 14037
imager = Imager(a,
                freq=lookingFreq,
                fieldRes = fieldRes)

imager.calculateRn()
imager.Rn_regularization = 1e-5*np.mean(np.abs(imager.Rn))
imager.Rx_regularization = 1e-2*np.mean(np.abs(imager.Rn))
t0 = time.time()
y_das = imager.beamform(#beamformer = "DAS")
                        beamformer = "X-KAT")
                        #beamformer = "CSM-KAT")
                        #beamformer = "MPDR")
                        #beamformer = "MVDR")
t1 = time.time()
print("Beamformer {} in {}s".format(imager.beamformer,t1-t0))

iterations = 2e2
t0 = time.time()
y = imager.DAMAS2(iterations = iterations,
                  #mode = "PSF")
                  #mode = "A-matrix")
                  mode = "KAT")
t1 = time.time()
print("DAMAS2 {}, {} iterations in {}s".format(imager.DAMAS2mode, imager.damas2Iterations,t1-t0))

plotArgs = {#'cmap':'jet',
            #'norm':LogNorm(),
            'interpolation':'none',
            'aspect':'equal'}

f.show(freq=imager.freq,
       windowFFT = windowFFT*f.sampleRate/a.sampleRate, # potencia do ruído depende do tamanho da janela
       **plotArgs)

fig, axes = plt.subplots(1, 2, figsize=(8,4))

z = imager.bfImg/np.max(imager.bfImg)
img = axes[0].imshow(z, **plotArgs)
plt.colorbar(img, ax=axes[0])
axes[0].set_title("Beamformer "+imager.beamformer+"\n"+str(imager.freq) + "Hz")

z = imager.damas2Img/np.max(imager.damas2Img)
img = axes[1].imshow(z, **plotArgs)
plt.colorbar(img, ax=axes[1])
axes[1].set_title("DAMAS2, "+str(imager.freq) +
                    "Hz\niter = " + str(imager.damas2Iterations))
