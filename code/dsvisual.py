import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
from ipywidgets import interact
from .stftprocessing import StftDataset
from tqdm import tqdm

class DatasetVisualizer(object):
    def __init__(self, examples, labels):
        if (not (isinstance(examples, StftDataset))) or (
            not (isinstance(labels, StftDataset))):
            print("Warning: Input not of type StftDataset")
        else:
            self.examples = examples
            self.labels = labels

            self.number_of_data = examples.get_data().shape[0]

            self.t = examples.t
            self.f = examples.f
        
        
    def plot_stft(self, index):
        amp_noisy = self.examples.get_data()[index,:,:,0]
        amp_pure = self.labels.get_data()[index,:,:,0]
        fig, [ax1, ax2] = plt.subplots(1,2)
        fig.set_size_inches(12,6)

        ax1.pcolormesh(self.t, self.f, amp_noisy, shading='gouraud')
        ax1.set_title('STFT Amplitude - Noisy')
        ax1.set_ylabel('Frequency [Hz]')
        ax1.set_xlabel('Time [sec]')
        
        ax2.pcolormesh(self.t, self.f, amp_pure, shading='gouraud')
        ax2.set_title('STFT Amplitude - Pure')
        ax2.set_xlabel('Time [sec]')
        plt.plot()
        
    
    def plot_time_domain(self, index):

        test_noisy = self.examples.get_data()[index,:,:,0]
        test_noisy_phase = self.examples.get_data()[index,:,:,1]

        test_pure = self.labels.get_data()[index,:,:,0]
        test_pure_phase = self.labels.get_data()[index,:,:,1]

        t, signal_pure = istft(test_pure*np.exp(1j*test_pure_phase), fs = 1./5.)
        t, signal_noisy = istft(test_noisy*np.exp(1j*test_noisy_phase), fs = 1./5.)

        plt.plot(t, signal_noisy)
        plt.plot(t, signal_pure, alpha=0.9)

    def browse_amplitude(self):
        interact(self.plot_stft, index=(0, self.number_of_data-1, 1))
        
    def browse_time_domain(self):
        interact(self.plot_time_domain, index=(0, self.number_of_data-1, 1))
    