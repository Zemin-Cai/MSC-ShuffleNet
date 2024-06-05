import os
import torchvision.transforms as trans
import cv2
import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageEnhance
import numpy as np
import random
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from skimage import transform,data,exposure
from PIL import Image

def random_rot(image, label):  # 翻转
    k = np.random.randint(-1, 2)
    image = np.rot90(image, k, (1, 2))
    label = np.rot90(label, k)
    return image, label

def random_flip(image, label):
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis+1).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):  # 旋转
    angle = np.random.randint(-15, 15)
    image = ndimage.rotate(image, angle, (2, 1), order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes,transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        mask=cv2.imread(os.path.join(self.mask_dir,img_id + self.mask_ext),0)
        #mask = np.where(mask == 3, 1, mask)
        #mask = np.where(mask == 4, 2, mask)

        img = img.astype('float32')

        img = img.transpose(2, 0, 1)
        sample = {'image': img, 'label': mask}
        if self.transform:
            img,mask = self.transform(sample)
        return img, mask,{'img_id': img_id}

