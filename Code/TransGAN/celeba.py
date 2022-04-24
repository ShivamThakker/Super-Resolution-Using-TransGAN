
from functools import partial
import torch
import os
import PIL
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity, verify_str_arg
from torch.utils.data import Dataset
import glob



class CelebA(Dataset):
    
    def __setup_files(self):
       
        file_names = os.listdir(self.data_dir)
        files = []  # initialize to empty list

        for file_name in file_names:
            possible_file = os.path.join(self.data_dir, file_name)
            if os.path.isfile(possible_file):
                files.append(possible_file)
                
        # return the files list
        return files

    def __init__(self, root, transform=None):
        
        self.data_dir = root
        self.transform = transform

        # setup the files for reading
        self.files = self.__setup_files()

    def __len__(self):
       
        return len(self.files)

    def __getitem__(self, idx):
       
        from PIL import Image

        # read the image:
        img_name = self.files[idx]
        if img_name[-4:] == ".npy":
            img = np.load(img_name)
            img = Image.fromarray(img.squeeze(0).transpose(1, 2, 0))
        else:
            img = Image.open(img_name)

        # apply the transforms on the image
        if self.transform is not None:
            img = self.transform(img)

        # return the image:
        return img, img
    
    
class FFHQ(Dataset):
   
    def __setup_files(self):
        
        file_names = glob.glob(os.path.join(self.data_dir, "./*/*.png")) + \
                     glob.glob(os.path.join(self.data_dir, "./*.jpg")) + \
                    [y for x in os.walk(self.data_dir) for y in glob.glob(os.path.join(x[0], "*.webp"))]
        files = []  # initialize to empty list

        for file_name in file_names:
            possible_file = os.path.join(self.data_dir, file_name)
            if os.path.isfile(possible_file):
                files.append(possible_file)

        # return the files list
        return files

    def __init__(self, root, transform=None):
       
        # define the state of the object
        self.data_dir = root
        self.transform = transform

        # setup the files for reading
        self.files = self.__setup_files()

    def __len__(self):
       
        return len(self.files)

    def __getitem__(self, idx):
        
        from PIL import Image

        # read the image:
        img_name = self.files[idx]
        if img_name[-4:] == ".npy":
            img = np.load(img_name)
            img = Image.fromarray(img.squeeze(0).transpose(1, 2, 0))
        else:
            img = Image.open(img_name)

        # apply the transforms on the image
        if self.transform is not None:
            img = self.transform(img)

        # return the image:
        return img, img
