from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from pathlib import Path

DATASET_ROOT = '../data/mapillary/'
GT_ROOT = './datasets/' # BECAREFUL, this is the ground truth that comes with GSV-Cities

class MSLSTest(Dataset):
    def __init__(self, input_transform = None):
        

        self.input_transform = input_transform

        self.dbImages = np.load(GT_ROOT+'msls_test/msls_test_dbImages.npy', allow_pickle=True)
        self.qImages = np.load(GT_ROOT+'msls_test/msls_test_qImages.npy', allow_pickle=True)
        
        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages))
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)
        self.ground_truth = None
    
    def __getitem__(self, index):
        img = Image.open(DATASET_ROOT + self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)
    
    def save_predictions(self, preds, path):
        with open(path, 'w') as f:
            for i in range(len(preds)):
                q = Path(self.qImages[i]).stem
                db = ' '.join([Path(self.dbImages[j]).stem for j in preds[i]])
                f.write(f"{q} {db}\n")
