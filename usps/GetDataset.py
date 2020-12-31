import numpy as np
import pandas as pd
import h5py
from sklearn import preprocessing
import scipy.io


class Dataset:
    def __init__(self, dataset_name):
        
        self.dataset_name = dataset_name
        

    def loadData(self):
        name = (self.dataset_name).lower()
        
        
        if name == 'usps':
            filename = 'usps.h5'
            f = h5py.File(filename, 'r')
            train_group = f['train']
            test_group = f['test']
            train_data=train_group['data'].value
            test_data=test_group['data'].value
            train_labels=train_group['target'].value
            test_labels=test_group['target'].value
            X=(np.concatenate((train_data,test_data)))
            labels=np.concatenate((train_labels,test_labels))
            f.close()
        
        else:
            print('Dataset not found')
            X=[]
            labels=[]
            
        return X,labels

        
        

