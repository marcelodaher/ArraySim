# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

from numpy import pi

def getU(fieldRes):
    U = np.zeros([fieldRes[0],fieldRes[1],3])
    U[:,:,0] = np.transpose(np.resize(
                np.linspace(-1,1,fieldRes[0]),[fieldRes[1],fieldRes[0]]))
    U[:,:,1] = np.resize(np.linspace(-1,1,fieldRes[1]),
                [fieldRes[0],fieldRes[1]])
    U[:,:,2] = np.sqrt(1-U[:,:,0]**2-U[:,:,1]**2)
    return U

class FarFieldSignal(object):

    """
     Incoming far field signal for the array
     

    :version:
    :author:
    """

    """ ATTRIBUTES

     fieldRes[0] -by- fieldRes[1] -by- N signal matrix.
     Signal[X0][Y0] is the N-length signal coming from the (X0,Y0) direction

    signal  (public)

     X resolution (pixels) for the far field signal

    fieldRes[0]  (public)

     Y resolution (pixels) for the far field signal

    fieldRes[1]  (public)

     Discrete-time N signal-length

    N  (public)

     Signal sample-rate in Hz

    sampleRate  (public)

    """
    def __init__(self, fieldRes, **kwargs):
        self.fieldRes = fieldRes
        self.nDirX = fieldRes[0]
        self.nDirY = fieldRes[1]
        self.nDir = np.product(fieldRes)
        
        if "N" in kwargs:
            self.N = kwargs['N']
        else:
            self.N = 1
                
        self.signal = np.zeros([fieldRes[0],fieldRes[1],self.N])
        self.noise = None
        
        if "sampleRate" in kwargs:
            self.sampleRate = kwargs['sampleRate']
            self.T = np.true_divide(1,self.sampleRate)
        else:
            self.sampleRate = None
            self.T = None
        
        '''
                  |sinϕ cosθ|
        U[x0,y0]= |sinϕ sinθ|
                  |  cosϕ   |
        '''
        self.U = getU(fieldRes)
        
    def addSource(self, signal, **kwargs):
        """
         Adds a new signal source to the Far Field 3d Signal

        @param undef signal : Signal array for source
        
        @param * _kwargs : Additional parameters for source addition.
Supports MxN matrix for 2d convolution
            must include "point" XOR "pattern" parameter
        @return  :
        @author
        """
        
        if self.sampleRate is None and "sampleRate" in kwargs:
            self.sampleRate  = kwargs["sampleRate"]
            del kwargs["sampleRate"]
        
        if "sampleRate" in kwargs:
            
            from scipy.interpolate import interp1d
            sampleRate = kwargs["sampleRate"]
            time_len = np.true_divide(len(signal),sampleRate)
            signal_time = np.true_divide(np.arange(len(signal)+1),sampleRate)
            new_signal_time = np.arange(0,time_len,1./self.sampleRate)
            interpolator = interp1d(signal_time,np.pad(signal,[0,1],'constant'),
                                    kind='cubic', fill_value = (0,0))
            signal = interpolator(new_signal_time)
            
        if len(signal) > self.N:
            self.N = len(signal)
            self.signal.resize([self.fieldRes[0],self.fieldRes[1],self.N])
            sizedSignal = signal
        else:
            sizedSignal = np.pad(signal,[0,self.N-len(signal)],'constant')
        
        newSignal = np.zeros([self.fieldRes[0],self.fieldRes[1],self.N])
        
        if "point" in kwargs:
            if np.shape(kwargs['point']) != (2,):
                raise TypeError("point must be 2-element array")
            x = kwargs['point'][0]
            y = kwargs['point'][1]
            newSignal[x][y] = sizedSignal
        elif "pattern" in kwargs:
            if np.shape(kwargs['pattern']) != (self.fieldRes[0], self.fieldRes[1]):
                raise TypeError("pattern must be MxN matrix")
            for x in range(self.fieldRes[0]):
                for y in range(self.fieldRes[1]):
                    newSignal[x,y] = np.multiply(kwargs['pattern'][x,y],sizedSignal)
        else:
            raise SyntaxError('missing "point" or "pattern" argument')
        
        self.signal = self.signal + newSignal
    
    def addNoise(self, variance = 1, mode = 'background', **kwargs):
        """
         Adds noise to the Far Field 3d Signal
        """
        
        if self.noise is None:
            self.noise= np.zeros([self.fieldRes[0],self.fieldRes[1],self.N])
        
        if self.sampleRate is None and "sampleRate" in kwargs:
            self.sampleRate  = kwargs["sampleRate"]
            del kwargs["sampleRate"]
        
        if mode == 'background':
            noise = np.random.normal(size = [self.fieldRes[0],self.fieldRes[1],self.N],
                                     scale = np.sqrt(variance))
            for i in range(self.fieldRes[0]):
                for j in range(self.fieldRes[1]):
                    if np.isnan(self.U[i,j,2]):
                        noise[i,j,:] = 0
        if mode == 'point':
            noise= np.zeros([self.fieldRes[0],self.fieldRes[1],self.N])
            x = kwargs['point'][0]
            y = kwargs['point'][1]
            if 'noise' not in kwargs:
                noise[x,y] = np.random.normal(size = self.N,
                                              scale = np.sqrt(variance))
            else:
                noise[x,y] = kwargs['noise']
        
        if 'bandwidth' in kwargs:
            '''
            bandwidth in Hz
            '''
            import scipy.signal as signal
            filt_num, filt_den = signal.butter(3, 2.*kwargs['bandwidth']/self.sampleRate)
            noise = signal.filtfilt(filt_num, filt_den, noise)
        
        self.noise = self.noise + noise
    
    def show(self, freq="all", ax=None, **kwargs):
        """
        Prints a heatmap of signal's power
        """
        
        if 'windowFFT' in kwargs:
            windowFFT = int(kwargs['windowFFT'])
            del kwargs['windowFFT']
            
        else:
            windowFFT = int(self.N)
        
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        if freq == "all":
            if self.noise is not None:
                img = np.sum(np.abs(np.fft.rfft(self.signal+self.noise))**2,axis=-1)
            else:
                img = np.sum(np.abs(np.fft.rfft(self.signal))**2,axis=-1)
            lookingFreq = "All Frequencies"
        else:
            signalFFT_len = int(np.ceil(np.true_divide(self.N, windowFFT)))
            if self.noise is not None:
                sinal_janelado = np.resize(self.signal+self.noise,[self.nDirX, self.nDirY,
                            signalFFT_len,windowFFT])
            else:
                sinal_janelado = np.resize(self.signal,[self.nDirX, self.nDirY,
                            signalFFT_len,windowFFT])
            freqs = np.fft.rfftfreq(windowFFT)*2*pi*self.sampleRate
            f_bin = 0
            freq = freq*2*pi #convertendo para rad/s
            while np.abs(freq-freqs[f_bin]) > np.abs(freq-freqs[f_bin+1]):
                f_bin = f_bin+1
            sinal_fft = np.mean(np.abs(np.fft.rfft(sinal_janelado)[:,:,:,f_bin]),axis=-1)
            img = np.power(sinal_fft,2)
            freqs = np.fft.rfftfreq(windowFFT)*self.sampleRate
            lookingFreq = "Freq = "+str(freqs[f_bin]) + "Hz"
        
        if 'interpolation' not in kwargs:
            kwargs['interpolation'] = 'none'
        if 'aspect' not in kwargs:
            kwargs['aspect'] = 'equal'
        img = ax.imshow(img/np.max(img),**kwargs)
        plt.colorbar(img, ax=ax)
        ax.set_title("Signal Field\n"+lookingFreq)
        
    def get_timeLength(self):
        self.timeLength = np.true_divide(self.N,self.sampleRate)
        return self.timeLength
