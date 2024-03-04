import numpy as np
from scipy.io import loadmat
import pathlib
import pickle
from scipy.signal import stft

class DataGenerator:
    def __init__(self, filelist, data_shape=np.array([29, 129]), seq_len = 20, shuffle=False, test_mode=False):

        # Init params
        self.shuffle = shuffle
        self.filelist = filelist
        self.data_shape = data_shape
        self.pointer = 0
        self.X = np.array([])
        self.y = np.array([])
        self.label = np.array([])
        self.boundary_index = np.array([])
        self.test_mode = test_mode

        self.seq_len = seq_len
        self.Ncat = 5
        # read from pkl file

        self.read_pkl_filelist(self.filelist)
        
        if self.shuffle:
            self.shuffle_data()

    def read_pkl_filelist(self,filelist):
        """
        Scan the file list and read them one-by-one
        """

        self.data_size = 0
        p = pathlib.Path(filelist)
        # filelist might be 'E:\\shhs\\polysomnography\\edfs\\processed\\pretext\\'

        files = list(p.rglob('*.pkl'))
        for f in files:
            sample = pickle.load(open(f, 'rb'))
            # sample['y'] is label
            self.data_size += len(sample['y']) # add number of epochs
            print(self.data_size)

        self.X = np.ndarray([self.data_size, self.data_shape[0], self.data_shape[1]])
        self.y = np.ndarray([self.data_size, self.Ncat])
        self.label = np.ndarray([self.data_size])
        count = 0
        for i in files:
            X, y, label = self.read_pkl_file(i)
            self.X[count : count + len(X)] = X
            self.y[count : count + len(X)] = y 
            self.label[count : count + len(X)] = label
            #self.boundary_index = np.append(self.boundary_index, [count, count +1, (count + len(X) - 1), (count + len(X) - 2)])
            self.boundary_index = np.append(self.boundary_index, np.arange(count, count + self.seq_len - 1))
            count += len(X)
            print(count)

        print("Boundary indices")
        print(self.boundary_index)
        self.data_index = np.arange(len(self.X))
        print(len(self.data_index))
        if self.test_mode == False:
            mask = np.in1d(self.data_index,self.boundary_index, invert=True)
            self.data_index = self.data_index[mask]
            #self.data_index = np.delete(self.data_index, self.boundary_index)
            print(len(self.data_index))
        print(self.X.shape, self.y.shape, self.label.shape)
        #self.data_size = len(self.label)

    def read_pkl_file(self,filename):
        # Load data
        print(filename)
        sample = pickle.load(open(filename, 'rb'))
        y = []
        label = []

        # deal with label in sleepEDF
        if 'cassette' in str(filename).split('\\')[-1]:
            # used for transform sleepEDF annotation
            # '1', '2', '3', '4', 'W', 'R'
            dic = dict()
            dic['W']=1
            dic['1']=2
            dic['2']=3
            dic['3']=4
            dic['4']=4
            dic['R']=5

            for l in sample['y']:
                if l not in ['W','1','2','3','4','R']:
                    y.append([0,0,0,0,0])
                    label.append(0)
                else:
                    label.append(dic[l])
                    if dic[l]==1:
                        y.append([1,0,0,0,0])
                    elif dic[l]==2:
                        y.append([0,1,0,0,0])
                    elif dic[l]==3:
                        y.append([0,0,1,0,0])
                    elif dic[l]==4:
                        y.append([0,0,0,1,0])
                    elif dic[l]==5:
                        y.append([0,0,0,0,1])
        else: # deal with shhs
            for l in sample['y']:
                if l<5: label.append(l+1)
                else: label.append(l)

                if l==0:
                    y.append([1,0,0,0,0])
                elif l==1:
                    y.append([0,1,0,0,0])
                elif l==2:
                    y.append([0,0,1,0,0])
                elif l==3:
                    y.append([0,0,0,1,0])
                elif l==5:
                    y.append([0,0,0,0,1])
                    
        X = []
        for L in range(len(sample['y'])):
            f, t, z = stft(sample['X'][0][3000*L:3000*(L+1)], fs=100, window='hamming', nfft=256, nperseg=200, boundary=None)
            z = np.array(z)
            z = np.abs(z)
            X.append(z.T)


        X = np.array(X)
        #X = np.transpose(X, (2, 1, 0))  # rearrange dimension
        y = np.array(y)
        #y = np.transpose(y, (1, 0))  # rearrange dimension
        label = np.array(label)
        #label = np.transpose(label, (1, 0))  # rearrange dimension
        #label = np.squeeze(label)

        return X, y, label

        #self.data_index = np.arange(len(self.X))
        #print(self.X.shape, self.y.shape, self.label.shape)
        #self.data_size = len(self.label)
        
    def shuffle_data(self):
        """
        Random shuffle the data points indexes
        """
        
        #create list of permutated index and shuffle data accoding to list
        #idx = np.random.permutation(len(x))
        idx = np.random.permutation(len(self.data_index))
        self.data_index = self.data_index[idx]

                
    def reset_pointer(self):
        """
        reset pointer to begin of the list
        """
        self.pointer = 0
        
        if self.shuffle:
            self.shuffle_data()
        
    
    def next_batch(self, batch_size):
        """
        This function gets the next n ( = batch_size) samples and labels
        """
        data_index = self.data_index[self.pointer:self.pointer + batch_size]

        #update pointer
        self.pointer += batch_size

        # after stack eeg, eog, emg, data_shape now is a 3D tensor
        batch_x = np.ndarray([batch_size, self.seq_len, self.data_shape[0], self.data_shape[1], self.data_shape[2]])
        batch_y = np.ndarray([batch_size, self.seq_len, self.y.shape[1]])
        batch_label = np.ndarray([batch_size, self.seq_len])

        for i in range(len(data_index)):
            for n in range(self.seq_len):
                batch_x[i, n]  = self.X[data_index[i] - (self.seq_len-1) + n, :, :, :]
                batch_y[i, n] = self.y[data_index[i] - (self.seq_len-1) + n, :]
                batch_label[i, n] = self.label[data_index[i] - (self.seq_len-1) + n]
            # check condition to make sure all corrections
            #assert np.sum(batch_y[i]) > 0.0

        # Get next batch of image (path) and labels
        batch_x.astype(np.float32)
        batch_y.astype(np.float32)
        batch_label.astype(np.float32)

        #return array of images and labels
        return batch_x, batch_y, batch_label

    # get and padding zeros the rest of the data if they are smaller than 1 batch
    # this necessary for testing
    def rest_batch(self, batch_size):

        #data_index = self.data_index[self.pointer : self.data_size]
        #actual_len = self.data_size - self.pointer
        data_index = self.data_index[self.pointer : len(self.data_index)]
        actual_len = len(self.data_index) - self.pointer

        # update pointer
        self.pointer = len(self.data_index)

        # after stack eeg, eog, emg, data_shape now is a 3D tensor
        batch_x = np.ndarray([actual_len, self.seq_len, self.data_shape[0], self.data_shape[1], self.data_shape[2]])
        batch_y = np.ndarray([actual_len, self.seq_len, self.y.shape[1]])
        batch_label = np.ndarray([actual_len, self.seq_len])

        for i in range(len(data_index)):
            for n in range(self.seq_len):
                batch_x[i, n]  = self.X[data_index[i] - (self.seq_len-1) + n, :, :, :]
                batch_y[i, n] = self.y[data_index[i] - (self.seq_len-1) + n, :]
                batch_label[i, n] = self.label[data_index[i] - (self.seq_len-1) + n]
            # check condition to make sure all corrections
            #assert np.sum(batch_y[i]) > 0.0

        # Get next batch of image (path) and labels
        batch_x.astype(np.float32)
        batch_y.astype(np.float32)
        batch_label.astype(np.float32)

        # return array of images and labels
        return actual_len, batch_x, batch_y, batch_label
