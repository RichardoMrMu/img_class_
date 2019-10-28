import torch
import os, glob
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset,DataLoader
from torchvision import  transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class HustData_LOAD(Dataset):
    def __init__(self,img, resize):
        super(HustData_LOAD, self).__init__()
        self.resize = resize
        self.images = self.load_img(img)


    def load_img(self,img):
        images = [img]
        return images


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idex):
        img = self.images[idex]
        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),

            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),

            transforms.RandomRotation(15),

            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img = tf(img)
        return img