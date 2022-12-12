""" Step (2) CUSTOMIZED AUGMENTATION
- Random Rotate
- Random Crop
- Add Noise (Gaussian) -> To Make the Augmented image pairs
- Cut Mix
"""
import random
import numpy as np
import torch
import math
import torch.nn.functional as F
from torchvision.transforms.functional import rotate


def random_crop(images: list, output_size: int):
  """ 8F 이미지 4개와 32F 이미지 1개에 모두 동일 위치에 대해서 random crop (32F 이미지를 crop은 무조건 해 주어야 한다.)
  - Anchor data와 positive pair data에 대해서 모두 적용이 된다.
  images: list of tensors
  output_size: crop patch의 한 변의 길이
  """
  H, W = images[0].shape[1:]
  new_h, new_w = output_size, output_size
  start_y = random.randint(0, H-new_h)
  start_x = random.randint(0, W-new_w)
  new_images = []
  for image in images:
    new_images.append(image[:,start_y:start_y + new_h, start_x:start_x + new_w])
  return new_images

def random_rotate(images: list):
  """ 8F 이미지 4개와 32F 이미지 1개를 모두 동일한 각도로 rotate 시켜 준다.
  """
  theta = random.randint(-180, 180)
  H, W = images[0].shape[1:]
  cosine, sine = math.cos(theta),math.sin(theta)
  new_h, new_w = round(abs(H*cosine)+abs(W*sine))+1, round(abs(W*cosine)+abs(H*sine))+1
  
  rotate_matrix = torch.Tensor(np.identity(images[0].shape[1]))
  rotate_matrix[:2,:2] = torch.Tensor(np.array([[math.cos(theta), -math.sin(theta)],
                                         [math.sin(theta), math.cos(theta)]]))
  
  org_center_x, org_center_y = round(((H+1)/2)-1), round(((W+1)/2)-1)
  new_center_x, new_center_y = round(((new_h+1)/2)-1), round(((new_w+1)/2)-1)

  new_images = []
  for image in images:
    theta = random.randint(-180, 180)
    new_images.append(rotate(image, angle = theta))
  return new_images
  
def cut_mix(images: list, mix_n = 7):
  """ 8F 이미지 4개와 32F 이미지에서 일정 비율로 이미지를 잘라서 붙여 준다.
  - 형체를 엉망으로 만들어 놓지만 근본적으로 갖고 있는 noise level은 일정하게 유지해 준다.
  """
  point = np.random.randint(1, 10)
  if point > 5:
    return images
  mix_size=images[0].shape[1] // 4
  center_x, center_y = images[0].shape[1] // 2, images[0].shape[2] // 2
  new_images = []
  for image in images:
    for n in range(mix_n):
      random_x, random_y = random.randint(0, images[0].shape[2] - mix_size), random.randint(0, images[0].shape[1] - mix_size)
      patch = image[:, random_y :random_y + mix_size, random_x: random_x + mix_size]
      random_x, random_y = random.randint(0, images[0].shape[2] - mix_size), random.randint(0, images[0].shape[1] - mix_size)
      print(random_x, random_y)
      image[:, random_y:random_y + mix_size, random_x: random_x + mix_size] = patch
    new_images.append(image)
  return new_images


def add_random_noise(images: list): 
  """ 8F 이미지 4개와 32F 이미지에서 Random 잡음을 추가해 준다.
  - 이는 positive sample끼리 공통적인 부분인 Noise Level을 최대한으로 모델이 발견할 수 있도록 하기 위함이다.
  """
  point = np.random.randint(1, 10)
  if point > 5:
    return images
  scale = np.random.randint(1, 50)
  random_noise = torch.Tensor(np.random.normal(loc=0.0, scale=float(scale), size=images[0].shape)).type(torch.float) ## 음수 ~ 양수 섞여 있음
  new_images = []
  random_noise = F.normalize(random_noise)
  for image in images:
    image = random_noise + image
    image = image.clamp(0,1)
    #image = F.normalize(image)
    new_images.append(image)
  return new_images




