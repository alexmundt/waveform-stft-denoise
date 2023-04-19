import os
import numpy as np

from scipy.signal import stft, istft

from ipywidgets import interact

from tqdm import tqdm

class StftDataset(object):
    """ A class to represent the STFT 
    processed datasets.
        Attributes
    ----------
    data : np.array
        STFT numpy array containing the processed files
        
    t : times of the STFT
    f : frequency bins of the STFT

    Methods
    -------
    process_data(data):
        data : np.array of waveforms
        
        processes the waveform into STFT form,
        stores the parameters of the STFT
        
    save(folder, name):
        saves the data in the given folder under
        the given name in a format used for
        the STFT dataclass
        
    load(folder, name):
        loads the data from the folder stored
        under the given name in a format used for
        the STFT dataclass
    
    """
    def __init__(self, data=None, fs = 1./5.):
        """ initialize
        """
        self.data = None
        if data is not None:
            self.process_data(data, fs=fs)
            
    def process_data(self, data, fs=1./5.):
        
        # process the data
        stfts = []
        for waveform in tqdm(data):
            f,t, result_stft = stft(waveform, fs = fs)
            stfts.append(result_stft)
            
        # store the axis parameters of the STFT (times and frequencies)
        self.f = f
        self.t = t

        # store the parameters of the 
        n = len(stfts)
        number_frequencies = stfts[0].shape[0]
        number_timeslices = stfts[0].shape[1]

        empty_data = np.zeros((n, number_frequencies, number_timeslices, 2))

        for count, current_stft in enumerate(stfts):
            amp = np.absolute(current_stft)
            phase = np.angle(current_stft)
            empty_data[count, :,:, 0] = amp
            empty_data[count, :,:, 1] = phase

        self.data = empty_data

    def save(self, folder, name):
        """ This methods saves the data under the name
        in the given folder
        """
        
        # check if data is actually available
        if self.data is None:
            print("No data stored yet.")
            return
        
        # create correct paths relative to the OS
        t_path = os.path.join(folder, name + "_t")
        f_path = os.path.join(folder, name + "_f")
        name_path = os.path.join(folder, name)
        
        # make a new folder if it already exists
        try: 
            os.mkdir(folder)
            print("Creating new folder...")
        except FileExistsError:
            print("Writing into existing folder...")
        
        # store the data
        np.save(t_path, self.t)
        np.save(f_path, self.f)
        np.save(name_path, self.data)
        
    def load(self, folder, name):
        """ This method loads the stored data (name) in 
        the given folder
        """
        # check if folder exists
        try:
            os.listdir(folder)
        except FileNotFoundError:
            print(f"Folder does not exist: {folder}")
            return

        # create correct paths relative to the OS
        t_path = os.path.join(folder, name + "_t.npy")
        f_path = os.path.join(folder, name + "_f.npy")
        name_path = os.path.join(folder, name +".npy")
        
        try:
            # load the data
            t = np.load(t_path)
            f = np.load(f_path)
            stfs = np.load(name_path)
            print("Files loaded.")
        except FileNotFoundError:
            print("Error: Files do not exist.")
            return
        
        self.t = t
        self.f = f
        self.data = stfs
            

    def get_data(self):
        return self.data