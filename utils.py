from PIL import Image
import os
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random
import torch
import numpy as np
from skimage import measure
from torch.nn import init

class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, cfg):
        super(TrainSetLoader, self).__init__()
        if isinstance(dataset_dir, str):
            self.dataset_dir = dataset_dir + '/patches_x' + str(cfg.scale_factor)
            self.file_list = os.listdir(self.dataset_dir)
            self.all_list = self.file_list
        
        elif isinstance(dataset_dir, list):
            k = len(dataset_dir)
            self.dataset_dir = []
            self.file_list = []
            self.all_list = []
            for i in range(k):
                temp = dataset_dir[i] + '/patches_x' + str(cfg.scale_factor)
                self.dataset_dir.append(temp)
                self.file_list = os.listdir(temp)
                for j in range(len(self.file_list)):
                    self.all_list.append(temp + '/' + self.file_list[j])
            
    def __getitem__(self, index):
    
        if isinstance(self.dataset_dir, str):
            img_hr_left  = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/hr0.png')
            img_hr_right = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/hr1.png')
            img_lr_left  = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/lr0.png')
            img_lr_right = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/lr1.png')

            img_hr_left  = np.array(img_hr_left,  dtype=np.float32)
            img_hr_right = np.array(img_hr_right, dtype=np.float32)
            img_lr_left  = np.array(img_lr_left,  dtype=np.float32)
            img_lr_right = np.array(img_lr_right, dtype=np.float32)

            img_hr_left, img_hr_right, img_lr_left, img_lr_right = augumentation(img_hr_left, img_hr_right, img_lr_left, img_lr_right)
            return toTensor(img_hr_left), toTensor(img_hr_right), toTensor(img_lr_left), toTensor(img_lr_right)
        
        elif isinstance(self.dataset_dir, list):    
            img_hr_left  = Image.open(self.all_list[index] + '/hr0.png')
            img_hr_right = Image.open(self.all_list[index] + '/hr1.png')
            img_lr_left  = Image.open(self.all_list[index] + '/lr0.png')
            img_lr_right = Image.open(self.all_list[index] + '/lr1.png')
            
            img_hr_left  = np.array(img_hr_left,  dtype=np.float32)
            img_hr_right = np.array(img_hr_right, dtype=np.float32)
            img_lr_left  = np.array(img_lr_left,  dtype=np.float32)
            img_lr_right = np.array(img_lr_right, dtype=np.float32)

            img_hr_left, img_hr_right, img_lr_left, img_lr_right = augumentation(img_hr_left, img_hr_right, img_lr_left, img_lr_right)
            return toTensor(img_hr_left), toTensor(img_hr_right), toTensor(img_lr_left), toTensor(img_lr_right)
            
    def __len__(self):
        return len(self.all_list)
    
class TrainSetLoaderMono(Dataset):
    def __init__(self, dataset_dir, cfg):
        super(TrainSetLoaderMono, self).__init__()
        self.scale_factor = cfg.scale_factor
        if isinstance(dataset_dir, str):
            self.dataset_dir = dataset_dir + '/patches_x' + str(cfg.scale_factor)
            self.file_list = os.listdir(self.dataset_dir)
            self.all_list = self.file_list
        
        elif isinstance(dataset_dir, list):
            k = len(dataset_dir)
            self.dataset_dir = []
            self.file_list = []
            self.all_list = []
            for i in range(k):
                temp = dataset_dir[i] + '/patches_x' + str(cfg.scale_factor)
                self.dataset_dir.append(temp)
                self.file_list = os.listdir(temp)
                for j in range(len(self.file_list)):
                    self.all_list.append(temp + '/' + self.file_list[j])
            
    def __getitem__(self, index):
    
        if isinstance(self.dataset_dir, str):
            img_hr_left  = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/hr0.png')
            img_lr_left  = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/lr0.png')

            img_lr_left = img_lr_left.resize((img_lr_left.size[0] * self.scale_factor, img_lr_left.size[1] * self.scale_factor),Image.BICUBIC)

            img_hr_left  = np.array(img_hr_left,  dtype=np.float32)
            img_lr_left  = np.array(img_lr_left,  dtype=np.float32)

            img_hr_left, img_lr_left = augumentation(img_hr_left, img_lr_left)
            return toTensor(img_hr_left), toTensor(img_lr_left)
        
        elif isinstance(self.dataset_dir, list):    
            img_hr_left  = Image.open(self.all_list[index] + '/hr0.png')
            img_lr_left  = Image.open(self.all_list[index] + '/lr0.png')
            
            img_lr_left = img_lr_left.resize((img_lr_left.size[0] * self.scale_factor, img_lr_left.size[1] * self.scale_factor),Image.BICUBIC)
            
            img_hr_left  = np.array(img_hr_left,  dtype=np.float32)
            img_lr_left  = np.array(img_lr_left,  dtype=np.float32)

            img_hr_left, img_lr_left = augumentation_mono(img_hr_left, img_lr_left)
            return toTensor(img_hr_left), toTensor(img_lr_left)
            
    def __len__(self):
        return len(self.all_list)


class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, scale_factor):
        super(TestSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        self.scale_factor = scale_factor
        self.file_list = os.listdir(os.path.join(dataset_dir, 'hr'))
    def __getitem__(self, index):
        hr_image_left  = Image.open(os.path.join(self.dataset_dir, 'hr', self.file_list[index], 'hr0.png'))
        hr_image_right = Image.open(os.path.join(self.dataset_dir, 'hr', self.file_list[index], 'hr1.png'))
        lr_image_left  = Image.open(os.path.join(self.dataset_dir, 'lr_x' + str(self.scale_factor), self.file_list[index], 'lr0.png'))
        lr_image_right = Image.open(os.path.join(self.dataset_dir, 'lr_x' + str(self.scale_factor), self.file_list[index], 'lr1.png'))
        hr_image_left  = ToTensor()(hr_image_left)
        hr_image_right = ToTensor()(hr_image_right)
        lr_image_left  = ToTensor()(lr_image_left)
        lr_image_right = ToTensor()(lr_image_right)
        return hr_image_left, hr_image_right, lr_image_left[:3, :, :], lr_image_right[:3, :, :]
    def __len__(self):
        return len(self.file_list)


class TestSetLoaderMono(Dataset):
    def __init__(self, dataset_dir, scale_factor):
        super(TestSetLoaderMono, self).__init__()
        self.dataset_dir = dataset_dir
        self.scale_factor = scale_factor
        self.file_list = os.listdir(os.path.join(dataset_dir, 'hr'))
    def __getitem__(self, index):
        hr_image_left  = Image.open(os.path.join(self.dataset_dir, 'hr', self.file_list[index], 'hr0.png'))
        lr_image_left  = Image.open(os.path.join(self.dataset_dir, 'lr_x' + str(self.scale_factor), self.file_list[index], 'lr0.png'))
        lr_image_left = lr_image_left.resize((lr_image_left.size[0] * self.scale_factor, lr_image_left.size[1] * self.scale_factor),Image.BICUBIC)
        hr_image_left  = ToTensor()(hr_image_left)
        lr_image_left  = ToTensor()(lr_image_left)
        return hr_image_left, lr_image_left[:3, :, :]
    def __len__(self):
        return len(self.file_list)



def augumentation(hr_image_left, hr_image_right, lr_image_left, lr_image_right):
        
        '''
        if random.random()<0.5:     # flip horizonly
            lr_image_left = lr_image_left[:, ::-1, :]
            lr_image_right = lr_image_right[:, ::-1, :]
            hr_image_left = hr_image_left[:, ::-1, :]
            hr_image_right = hr_image_right[:, ::-1, :]
        '''    
           
        if random.random()<0.5:     #flip vertically
            lr_image_left = lr_image_left[::-1, :, :]
            lr_image_right = lr_image_right[::-1, :, :]
            hr_image_left = hr_image_left[::-1, :, :]
            hr_image_right = hr_image_right[::-1, :, :]

        return np.ascontiguousarray(hr_image_left), np.ascontiguousarray(hr_image_right), \
                np.ascontiguousarray(lr_image_left), np.ascontiguousarray(lr_image_right)

def augumentation_mono(hr_image_left, lr_image_left):
        
        '''
        if random.random()<0.5:     # flip horizonly
            lr_image_left = lr_image_left[:, ::-1, :]
            lr_image_right = lr_image_right[:, ::-1, :]
            hr_image_left = hr_image_left[:, ::-1, :]
            hr_image_right = hr_image_right[:, ::-1, :]
        '''    
           
        if random.random()<0.5:     #flip vertically
            lr_image_left = lr_image_left[::-1, :, :]
            hr_image_left = hr_image_left[::-1, :, :]

        return np.ascontiguousarray(hr_image_left), np.ascontiguousarray(lr_image_left)


def toTensor(img):
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255)

class L1Loss(object):
    def __call__(self, input, target):
        return torch.abs(input - target).mean()

def cal_psnr(img1, img2):
    img1_np = np.array(img1.cpu())
    img2_np = np.array(img2.cpu())

    return measure.compare_psnr(img1_np, img2_np)

def save_ckpt(state, save_path='log', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal(m.weight.data)
        

def cal_ssim(img1, img2):
    img1 = img1.permute(0,2,3,1)
    img2 = img2.permute(0,2,3,1)
    
    img1_np = np.array(img1.cpu())
    img2_np = np.array(img2.cpu())
    img1_np = np.squeeze(img1_np, axis = 0)
    img2_np = np.squeeze(img2_np)
    
    return measure.compare_ssim(img1_np, img2_np, multichannel = True)
