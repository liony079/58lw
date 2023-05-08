import torch
import torch.utils.data as data
import numpy as np
import time
import scipy
import spectrum
import scipy
from sklearn import preprocessing
import pandas as pd
import os
import pickle
from scipy import stats
import matplotlib.pyplot as plt
import scipy.io as scio

def load_training(windowsize, period, stride, subject, batchsize, shuffle, pinmemory, numworkers, droplast):
    dataset = Ninaprodataset(
                             windowsize=windowsize,
                             period=period,
                             subject=subject,
                             stride=stride
                              )
    train_loader = data.DataLoader(dataset, batch_size=batchsize, shuffle=shuffle, num_workers=numworkers, pin_memory=pinmemory, drop_last=droplast)
    return train_loader

def load_testing(windowsize, period, stride, subject, batchsize,  pinmemory, numworkers, droplast):
    dataset = Ninaprodataset(
                             windowsize=windowsize,
                             period=period,
                             subject=subject,
                             stride=stride,
                              )
    test_loader = data.DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=numworkers, pin_memory=pinmemory, drop_last=droplast)
    return test_loader

class Ninaprodataset(data.Dataset):

    def __init__(self, windowsize, subject, period,stride):
        semg_set = np.load("C:\\Users\\Administrator\\Desktop\\sEMG_database\\s"+str(subject)+"\\E2\\emg.npy")
        semg_set = semg_set[int(semg_set.__len__()*period[0]) : int(semg_set.__len__()*period[1]),1]
        semg_set = (semg_set - np.mean(semg_set))/np.std(semg_set)
        # semg_set = semg_set[:semg_set.size//1024*1024].reshape(1024,-1)
        semg_set = np.lib.stride_tricks.as_strided(semg_set,
                                        shape=((len(semg_set) - windowsize) // stride + 1, windowsize),
                                        strides=(stride * 8, 8))
        self.orgframe = semg_set[:,np.newaxis]
        #waveletbasis = scio.loadmat('C:\\Users\\Administrator\\Desktop\\sEMG-reconstructure\\sEMG-reconstructure\\coif5wavelets.mat')["ww"]
        waveletbasis =scio.loadmat('C:\\Users\\Administrator\\Desktop\\sEMG-reconstructure\\sEMG-reconstructure\\db2wavelets.mat')["ww"]
        # Phi = np.random.randn(int(windowsize * 0.9), windowsize)
        Phi = np.load('C:\\Users\\Administrator\\Desktop\\sEMG-reconstructure\\sEMG-reconstructure\\Phi.npy')
        transmit = np.dot(Phi, waveletbasis)
        measurement = np.dot(transmit, semg_set.transpose())
        self.compframe = measurement.transpose()[:,np.newaxis]


    def __getitem__(self, index):

        orgframe = self.orgframe[index,:,:]
        compframe = self.compframe[index,:,:]
        return orgframe, compframe

    def __len__(self):

        return self.compframe.shape[0]


if __name__ == "__main__":
    trainloader = load_training(
                                windowsize=1024,
                                subject=1,
                                period=[0,0.8],
                                stride=200,
                                batchsize=16,
                                shuffle=False,
                                pinmemory=True,
                                numworkers=8,
                                # numworkers=0,
                                droplast=True)

    testloader = load_testing(
                                windowsize=1024,
                                stride=200,
                                period=[0.8, 1],
                                subject=1,
                                batchsize=16,
                                pinmemory=True,
                                numworkers=16,
                                # numworkers=0,
                                droplast=True)

    print(trainloader.__len__())
    print(testloader.__len__())

    clc=[]
    for step,(inputs,classes) in enumerate(trainloader):

        print('-----------------------------------')
        print(step)
        print(inputs.shape)
        print(classes.shape)

        print(inputs.max().data)
        print(inputs.min().data)
        break
