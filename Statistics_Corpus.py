import numpy as np
import  torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd

import os



def DataLoder(path_tt):
    path = path_tt
    data = pd.read_csv(path)
    text = data['Text'].tolist()
    image = data['Image'].tolist()
    label = data['ReLabel'].tolist()

    total_w = {}
    unique_w = {}
    for t in text:
        t = t.split()
        for w in t:
            w = w.replace(' ','')
            if w in total_w:
                total_w[w] += 1
                print(w)
            else:
                total_w[w] = 1
                unique_w[w] = 1
    print('Total Words:',len(total_w))
    print('Unique Words:',len(unique_w))

if __name__ == '__main__':
    #DataLoder('/media/tigerit/ssd/MMSA/Multimodal-Sentiment/train.csv')
    DataLoder('/media/tigerit/ssd/MMSA/Multimodal-Sentiment/test.csv')