# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Circle
from scipy.interpolate import interp1d

from sys import version_info
if(version_info >= (3,0)):
    from importlib import reload as reload
    
import FarFieldSignal
reload(FarFieldSignal)
from FarFieldSignal import *

#import Imager
#reload(Imager)

class GridArray(object):

    """
     A sensor array arranged in a grid format

    :version:
    :author:
    """

    """ ATTRIBUTES

     X coordinates where there are sensors

    microphonesX  (public)

     Y coordinates where there are sensors

    microphonesY  (public)

     Signal sampling rate in Hz

    sampleRate  (public)

     Resolution in which the samples are stored.

    resolution  (public)

    """
    def __init__(self,microphonesX,microphonesY,
                 resolution = np.dtype(np.float64),
                 **kwargs):

        self.microphonesX = np.sort(microphonesX
            - np.average([min(microphonesX),max(microphonesX)]))
        self.microphonesY = np.sort(microphonesY
            - np.average([min(microphonesY),max(microphonesY)]))
        
        self.resolution = resolution
        
        if "sampleRate" in kwargs:
            self.sampleRate = np.float(kwargs["sampleRate"])
        else:
            self.sampleRate = None
        
        if "c" in kwargs:
            self.c = np.float(kwargs["c"])
        else:
            self.c = 343 # m/s, sound speed
            
        if "verbose" in kwargs:
            self.verbose = kwargs['verbose']
        else:
            self.verbose = False
        
        if len(microphonesX)>1:
            if len(microphonesY)>1:
                minD = np.min(np.abs([microphonesX[:-1]-microphonesX[1:],
                    microphonesY[:-1]-microphonesY[1:]]))
            else: 
                minD = np.min(np.abs(microphonesX[:-1]-microphonesX[1:]))
        elif len(microphonesY)>1:
            minD = np.min(np.abs(microphonesY[:-1]-microphonesY[1:]))

        #self.fc = self.c/minD
        
        self.signal = None
        self.noise = None
        self.signalFFT = None
        self.noiseFFT = None
        self.windowFFT = None
        
        self.nMicX = len(self.microphonesX)
        self.nMicY = len(self.microphonesY)
        self.nMic = self.nMicX*self.nMicY
        
    def show(self, ax = None, fig=None,
             title = "Configuracao do arranjo",
             zero='center'):
        
        if zero == 'center':
            microphonesX = self.microphonesX
            microphonesY = self.microphonesY
        elif zero == 'zero':
            microphonesX = self.microphonesX - min(self.microphonesX)
            microphonesY = self.microphonesY - min(self.microphonesY)
        
        rad = 0.2*min(np.concatenate([microphonesX[1:]-microphonesX[:-1],
                       microphonesY[1:]-microphonesY[:-1]]))
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        for x in microphonesX:
            ax.axvline(x, linestyle = '--', color = 'black',linewidth=0.5,alpha=0.5)
            for y in microphonesY:
                ax.add_patch(Circle((x,y),rad, color='red'))
        for y in microphonesY:
            ax.axhline(y, linestyle = '--', color = 'black',linewidth=0.5,alpha=0.5)
        #plt.tight_layout()
        #fig.autofmt_xdate()
        ax.autoscale()
        ax.axis('equal')
        ax.set_xticks(microphonesX)
        ax.set_yticks(microphonesY)
        ax.set_title(title)
        ax.set_xlabel("x[m]")
        ax.set_ylabel("y[m]")
        
    def calculateDelays(self, farField):
        
        delays = np.zeros([len(self.microphonesX), len(self.microphonesY),
                    farField.fieldRes[0], farField.fieldRes[1]],
                    dtype = 'float64')
        
        # para cada microfone
        for x in range(len(self.microphonesX)):
            for y in range(len(self.microphonesY)):
                # calculamos o atraso para cada fonte
                delays[x,y] = -np.dot(farField.U,[self.microphonesX[x],
                                        self.microphonesY[y],0])/self.c
        
        return delays
        
    def receiveNoise(self, incomingSignal):
        """
         

        @param FarFieldSignal incomingSignal : Far field signal incoming to the sensor array
        @return  :
        @author
        """
        if self.verbose: print("Receiving the Far Field Noise")
        delays = self.calculateDelays(incomingSignal)
                            
        self.N = np.ceil(np.true_divide(incomingSignal.N*self.sampleRate,
                                        incomingSignal.sampleRate))
        self.timeLength = np.true_divide(self.N,self.sampleRate)
        timeSeries = np.arange(self.N)*np.true_divide(1,self.sampleRate)
        
        self.noise = np.zeros([self.nMicX, self.nMicY, int(self.N)],
                                dtype = self.resolution)
        pad_size = 1000
        pad_time = np.true_divide(pad_size,incomingSignal.sampleRate)
        incomingSignal.get_timeLength()
        
        # para cada pixel de sinal recebido
        for px in range(incomingSignal.fieldRes[0]):
            for py in range(incomingSignal.fieldRes[1]):
                if (incomingSignal.noise[px,py]!=0).any():
                    padded_signal = np.pad(incomingSignal.noise[px,py],
                                        pad_size,'constant')
                    incomingSignal_time = np.linspace(-pad_time,
                                        incomingSignal.timeLength + pad_time,
                                        incomingSignal.N+2*pad_size,
                                        endpoint=False)
                    interpolator = interp1d(incomingSignal_time,padded_signal,
                                            kind='cubic', fill_value = (0,0))
                    for mx in range(len(self.microphonesX)):
                        for my in range(len(self.microphonesY)):
                            d = delays[mx,my,px,py]
                            if np.isnan(d) or np.isinf(d):
                                pass
                            else:
                                delayed_time = timeSeries + delays[mx,my,px,py]
                                self.noise[mx,my] += interpolator(delayed_time)
            
        self.noiseFFT = None
        if self.verbose: print("Received noise DONE")
        return
        
    def receiveSignal(self, incomingSignal):
        """
         

        @param FarFieldSignal incomingSignal : Far field signal incoming to the sensor array
        @return  :
        @author
        """
        if self.verbose: print("Simulating the received signal")
        delays = self.calculateDelays(incomingSignal)
                            
        self.N = np.ceil(np.true_divide(incomingSignal.N*self.sampleRate,
                                        incomingSignal.sampleRate))
        self.timeLength = np.true_divide(self.N,self.sampleRate)
        timeSeries = np.arange(self.N)*np.true_divide(1,self.sampleRate)
        
        self.signal = np.zeros([self.nMicX, self.nMicY, int(self.N)],
                                dtype = self.resolution)
        pad_size = 1000
        pad_time = np.true_divide(pad_size,incomingSignal.sampleRate)
        incomingSignal.get_timeLength()
        
        # para cada pixel de sinal recebido
        for px in range(incomingSignal.fieldRes[0]):
            for py in range(incomingSignal.fieldRes[1]):
                if (incomingSignal.signal[px,py]!=0).any():
                    padded_signal = np.pad(incomingSignal.signal[px,py],
                                        pad_size,'constant')
                    incomingSignal_time = np.linspace(-pad_time,
                                        incomingSignal.timeLength + pad_time,
                                        incomingSignal.N+2*pad_size,
                                        endpoint=False)
                    interpolator = interp1d(incomingSignal_time,padded_signal,
                                            kind='cubic', fill_value = (0,0))
                    for mx in range(len(self.microphonesX)):
                        for my in range(len(self.microphonesY)):
                            d = delays[mx,my,px,py]
                            if np.isnan(d) or np.isinf(d):
                                pass
                            else:
                                delayed_time = timeSeries + delays[mx,my,px,py]
                                self.signal[mx,my] += interpolator(delayed_time)
        
        if incomingSignal.noise is not None:
            self.receiveNoise(incomingSignal)
            self.signal += self.noise
        self.signalFFT = None
        if self.verbose: print("Received signal DONE")
        return
    
    def receiveSignal2(self, incomingSignal):
        pass
        if self.sampleRate != incomingSignal.sampleRate:
            N0 = incomingSignal.signal.shape[-1]
            N1 = np.floor_divide(self.sampleRate*N0,incomingSignal.sampleRate)
            time0 = np.linspace(0,1,num=N0)
            time1 = np.linspace(0,1,num=N1)
            inSignal = np.zeros([incomingSignal.fieldRes[0],incomingSignal.fieldRes[1],N1])
            for px in range(incomingSignal.fieldRes[0]):
                for py in range(incomingSignal.fieldRes[1]):
                    if (incomingSignal.signal[px,py]!=0).any():
                        interpolator = interp1d(time0,incomingSignal.signal[px,py],
                                                kind='cubic', fill_value = (0,0))
                        inSignal[px,py] = interpolator(time1)
        else:
            N1 = N0 = incomingSignal.signal.shape[-1]
            inSignal = incomingSignal.signal
        freqs = np.fft.rfftfreq(N1, d = 1./self.sampleRate)
        inSignalFFT = np.fft.rfft(inSignal)
        signalFFT = np.zeros([self.nMicX,self.nMicY,1,len(freqs)],dtype = np.complex128)
        for i in range(len(freqs)):
            freq = freqs[i]
            dx = np.linspace(-1,1,incomingSignal.fieldRes[0])
            Vx = (np.reshape(self.microphonesX,[self.nMicX,1])).dot(np.reshape(dx,[incomingSignal.fieldRes[0],1]).T)
            Vx = np.exp(-1j*Vx*2*np.pi*freq/self.c)
            dy = np.linspace(-1,1,incomingSignal.fieldRes[1])
            Vy = (np.reshape(self.microphonesY,[self.nMicY,1])).dot(np.reshape(dy,[incomingSignal.fieldRes[1],1]).T)
            Vy = np.exp(-1j*Vy*2*np.pi*freq/self.c)
            signalFFT[:,:,0,i] = Vy.dot(inSignalFFT[:,:,i]).dot(Vx.T)
        
        self.receiveNoise2(incomingSignal)
        
        self.signal = np.fft.irfft(signalFFT) + self.noise
        self.signalFFT = np.fft.rfft(self.signal)
        self.windowFFT = N1
        self.N = N1
        return
            
    def receiveNoise2(self, incomingSignal):
        pass
        if self.sampleRate != incomingSignal.sampleRate:
            N0 = incomingSignal.noise.shape[-1]
            N1 = np.floor_divide(self.sampleRate*N0,incomingSignal.sampleRate)
            time0 = np.linspace(0,1,num=N0)
            time1 = np.linspace(0,1,num=N1)
            inSignal = np.zeros([incomingSignal.fieldRes[0],incomingSignal.fieldRes[1],N1])
            for px in range(incomingSignal.fieldRes[0]):
                for py in range(incomingSignal.fieldRes[1]):
                    if (incomingSignal.noise[px,py]!=0).any():
                        interpolator = interp1d(time0,incomingSignal.noise[px,py],
                                                kind='cubic', fill_value = (0,0))
                        inSignal[px,py] = interpolator(time1)
        else:
            N1 = N0 = incomingSignal.noise.shape[-1]
            inSignal = incomingSignal.noise
        freqs = np.fft.rfftfreq(N1, d = 1./self.sampleRate)
        inSignalFFT = np.fft.rfft(inSignal)
        signalFFT = np.zeros([self.nMicX,self.nMicY,1,len(freqs)],dtype = np.complex128)
        for i in range(len(freqs)):
            freq = freqs[i]
            Vx = np.zeros([self.nMicX,incomingSignal.fieldRes[0]], dtype = 'complex128')
            dx = np.linspace(-1,1,incomingSignal.fieldRes[0])
            for x in range(self.nMicX):
                Vx[x] = dx*self.microphonesX[x]/self.c
            Vx = np.exp(-1j*Vx*2*np.pi*freq)
            Vy = np.zeros([self.nMicY,incomingSignal.fieldRes[1]], dtype = 'complex128')
            dy = np.linspace(-1,1,incomingSignal.fieldRes[1])
            for y in range(self.nMicY):
                Vy[y] = dy*self.microphonesY[y]/self.c
            Vy = np.exp(-1j*Vy*2*np.pi*freq)
            signalFFT[:,:,0,i] = Vy.dot(inSignalFFT[:,:,i]).dot(Vx.T)
        self.noiseFFT = signalFFT
        self.noise = np.fft.irfft(signalFFT)
        return
        
    def saveMicSignal(self,**kwargs):
        if 'fname' not in kwargs:
            kwargs['fname'] = 'mic_signal.csv'
        if 'delimiter' not in kwargs:
            kwargs['delimiter'] = ';'
        kwargs['X'] = np.reshape(self.signal,[self.nMic,self.N])
        np.savetxt(**kwargs)
    
    def playMic(self, micPosition):
        from scipy.io import wavfile
        import winsound
        
        micPosition = tuple(micPosition)
        wavfile.write('temp.wav',self.sampleRate,np.array(self.signal[micPosition]
                            ,dtype = np.float32))
        winsound.PlaySound('temp.wav',winsound.SND_ASYNC)
    
    def addNoise(self, snr):
        if self.verbose: print("Adding sampling noise")
        signalPower = np.mean(self.signal**2)
        self.noiseVar = signalPower/np.power(10,0.1*snr)
        noise = np.random.normal(scale = np.sqrt(self.noiseVar),
                                size = np.shape(self.signal))
        self.signal = self.signal + noise
        #if self.noise is not None:
        #    self.noise += noise
    
    def noise2FFT(self, **kwargs):
        if self.verbose: print("Converting noise to frequency domain")
        if self.noise.any() == None:
            raise TypeError("Received noise must be defined")
        if 'windowFFT' in kwargs:
            self.windowFFT = int(kwargs['windowFFT'])
        else:
            if self.windowFFT is None: self.windowFFT = int(self.N)
        
        # janelando o sinal para realizar a DTFT
        signalFFT_len = int(np.ceil(np.true_divide(self.N,self.windowFFT)))
        sinal_janelado = np.resize(self.noise,[self.nMicX, self.nMicY,
                            signalFFT_len,int(self.windowFFT)])
        # realizando a fft pós janelamento
        self.noiseFFT = np.fft.rfft(sinal_janelado)
    
    def signal2FFT(self, **kwargs):
        
        if self.noise is not None:
            self.noise2FFT(**kwargs)
        if self.verbose: print("Converting Signal to frequency domain")
        if self.signal.any() == None:
            raise TypeError(" Received signal must be defined")
        if 'windowFFT' in kwargs:
            self.windowFFT = int(kwargs['windowFFT'])
        else:
            if self.windowFFT is None: self.windowFFT = int(self.N)
        
        # janelando o sinal para realizar a DTFT
        signalFFT_len = int(np.ceil(np.true_divide(self.N,self.windowFFT)))
        sinal_janelado = np.resize(self.signal,[self.nMicX, self.nMicY,
                            signalFFT_len,int(self.windowFFT)])
        # realizando a fft pós janelamento
        self.signalFFT = np.fft.rfft(sinal_janelado)