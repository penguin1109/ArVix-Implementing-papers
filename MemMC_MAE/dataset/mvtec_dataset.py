import glob
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import os
import cv2

class MVTecDataset(Dataset):
  def __init__(self, cfg, mode='train'):
    super(MVTecDataset, self).__init__()
    self.cfg = cfg
    self.mode = mode
    self.base_path = '/content/drive/MyDrive/Competition/dacon/anomal_classify/mvtec'
    self._make_file_dir()

  def _make_file_dir(self):
    classes = sorted(os.listdir(self.base_path))
    self.files = []
    for cls in classes:
      files = glob.glob(self.base_path + f"/{cls}/{self.mode}/good/*.png")
      self.files.extend(files)
 
  def __len__(self):
    return len(self.files)
  
  def _augment(self):
    aug = []
    aug.append(transforms.ToTensor())
    if self.mode == 'train':
      aug.append(transforms.RandomResizedCrop(self.cfg.img_size, scale = (0.2, 1.0), interpolation=3))
      aug.append(transforms.RandomHorizontalFlip(0.5))
    else:
      aug.append(transforms.Resize(self.cfg.img_size))
    aug.append(transforms.Normalize(self.cfg.mean, self.cfg.std))
    aug = transforms.Compose(aug)

    return aug

  def __getitem__(self, idx):
    fname = self.files[idx]
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    aug = self._augment()
    img = aug(img)
    img = img.to(dtype = torch.float)

    return img

    
    

    
