import numpy as np
import math
from tensorflow.keras.utils import Sequence
import tensorflow as tf

class DataLoader(Sequence):
    """Custom DataLoader
    Args:
        dataset (Sequence): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
        suffle (bool, optional): set to True to have the data reshuffled at every epoch
    """

    def __init__(self, dataset, batch_size=8, shuffle=False, is_train=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.is_train = is_train
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return math.ceil(len(self.dataset) / self.batch_size)

    def __getitem__(self, idx):
        'Generates one batch of the dataset'
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_data = self.__data_generation(indices)
        return batch_data

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __data_generation(self, indices):
        'Generates data containing batch_size samples'
        batch_data = np.array([self.dataset[i] for i in indices])
        if self.is_train:
            batch_data = [np.stack(batch_data[:,i],axis=0) for i in range(4)] 
        return batch_data