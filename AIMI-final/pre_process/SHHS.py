#import sys
import mne
import numpy as np
import os
#import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import pathlib
from multiprocessing import Process
import pickle
import argparse

def pretext_train_test(root_folder, k, N, epoch_sec):
    # get all data indices
    p = pathlib.Path(root_folder+'label\\')
    files = list(p.rglob('*.xml'))
    all_index = []
    for f in files:
        end_f = str(f).split('\\')[-1]
        if not os.path.exists(root_folder + 'processed\\pretext\\shhs1-' + str(end_f.split('-')[1])+'.pkl'):
            all_index.append(int(end_f.split('-')[1]))
    
    # split into 
    # pretext_index = np.random.choice(all_index, int(len(all_index) * 0.98), replace=False)
    # train_index = np.random.choice(list(set(all_index) - set(pretext_index)), int(len(all_index) * 0.01), replace=False)
    # test_index = list(set(all_index) - set(pretext_index) - set(train_index))

    print ('start pretext process!')
    sample_process(root_folder, k, N, epoch_sec, 'pretext', all_index)
    print ()
    
    # print ('start train process!')
    # sample_process(root_folder, k, N, epoch_sec, 'train', train_index)
    # print ()
    
    # print ('start test process!')
    # sample_process(root_folder, k, N, epoch_sec, 'test', test_index)
    # print ()


def sample_process(root_folder, k, N, epoch_sec, train_test_val, index):
    # process each EEG sample: further split the samples into window sizes and using multiprocess
    for i, j in enumerate(index):
        if i % N == k:
            if k == 0:
                print ('Progress: {} / {}'.format(i, len(index)))

            # load the signal "X" part
            data = mne.io.read_raw_edf(root_folder + 'shhs1\\' + 'shhs1-' + str(j) + '.edf')
            data.resample(100)
            X = data.get_data()
            
            # some EEG signals have missing channels, we treat them separately
            # if X.shape[0] == 16:
            #     X = X[[0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15], :]
            # elif X.shape[0] == 15:
            #     X = X[[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14], :]
            
            # only C4-A1 channel
            X = X[mne.pick_channels(data.ch_names, ['EEG']), :]

            # load the label "Y" part
            with open(root_folder + 'label\\' + 'shhs1-' + str(j) + '-profusion.xml', 'r') as infile:
                text = infile.read()
                root = ET.fromstring(text)
                y = [int(i.text) for i in root.find('SleepStages').findall('SleepStage')]

            # slice the EEG signals into non-overlapping windows, window size = sampling rate per second * second time = 100 * windowsize
            path = root_folder + 'processed\\{}\\'.format(train_test_val) + 'shhs1-' + str(j) + '.pkl'
            pickle.dump({'X': X, 'y': y}, open(path, 'wb'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--windowsize', type=int, default=30, help="unit (seconds)")
    parser.add_argument('--multiprocess', type=int, default=1, help="How many processes to use")
    args = parser.parse_args()

    root_folder = 'E:\\shhs\\polysomnography\\edfs\\'

    if not os.path.exists(root_folder+'processed\\'):
        os.makedirs(root_folder+'processed\\pretext')
        #os.makedirs(root_folder+'SHHS_data/processed/train')
        #os.makedirs(root_folder+'SHHS_data/processed/test')
    
    N, epoch_sec = args.multiprocess, args.windowsize
    p_list = []
    for k in range(N):
        process = Process(target=pretext_train_test, args=(root_folder, k, N, epoch_sec))
        process.start()
        p_list.append(process)

    for i in p_list:
        i.join()
