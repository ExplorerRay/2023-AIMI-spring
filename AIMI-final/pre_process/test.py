import pickle
from scipy.signal import stft
import numpy as np

path = 'E:\\shhs\\polysomnography\\edfs\\processed\\pretext\\shhs1-200492.pkl'

sample = pickle.load(open(path, 'rb'))


print(len(sample['X'][0]))

print(sample['y'])

X = []
for L in range(len(sample['y'])):
    f, t, z = stft(sample['X'][0][3000*L:3000*(L+1)], fs=100, window='hamming', nfft=256, nperseg=200, boundary=None)
    z = np.array(z)
    X.append(z.T)
    #print(t)

X = np.array(X)
print(X.shape) # (number of 30s-epochs, 29, 129)
