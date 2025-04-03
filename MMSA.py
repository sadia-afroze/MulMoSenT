import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def CSVwRiter():
    ImagePath = './SentimentImages/'
    TextPath = './SentimentText/'
    with open('./MMSA.csv', 'a', encoding='utf-8') as f:
        f.write(f'Image,Text,Label\n')
    for folder in os.listdir(ImagePath):
        folderPath = os.path.join(ImagePath, folder)
        print(f"Folder Name {folder}")
        for file in os.listdir(folderPath):
            #File name without extension
            root = os.path.splitext(file)[0]
            textPath = os.path.join(TextPath, folder, root+'.txt')
            ImgPath = os.path.join(folderPath, file)
            text=''
            with open(textPath, 'r', encoding='utf-8') as f:
                text = str(f.read())
            text = text.replace('\n', ' ')
            text = text.replace(',', ' ')
            text = text.replace('"', ' ')
            text = text.replace("\t", ' ')
            with open('./MMSA.csv', 'a', encoding='utf-8') as f:
                f.write(f'{ImgPath},{text},{folder}\n')



if __name__=='__main__':
    data = pd.read_csv('./MMSA.csv')
    Text = data['Text']
    ImagePath = data['Image']
    Labels = data['Label']
    print(data.head())