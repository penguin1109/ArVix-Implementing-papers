""" Step (3) MAKING THE DATASET
- 앞서 SupervisedClassMaker 객체를 사용해서 구한 unique한 class들의 정보를 사용해서 
"""
import torch.utils.data as data
from torchvision import transforms
import cv2
from PIL import Image
import torch
import os
import numpy as np
from .augment import random_crop, random_rotate, add_random_noise, cut_mix
from .class_maker import SupervisedClassMaker
import numpy as np

## 기존의 AIMS_MULITFRAME_DATASET과 동일한 argument들을 입력으로 넣어 주면 되기 때문에 별도의 처리가 entry 파일에서 필요가 없다.
class AIMSDataset(data.Dataset):
    def __init__(self, 
               asset_configuration,
               data_configurations,
               parameter_mappings,
               meta_datas,
               mode="train"):
        super(AIMSDataset, self).__init__()
        self.mode = mode
        self.crop_size = 400
        self.normalize_std = 0.5
        self.normalize_mean = 0.0
        class_maker = SupervisedClassMaker(asset_configuration, data_configurations, parameter_mappings, meta_datas)
        self.image_paths = class_maker.image_paths
        self.class_dict = class_maker.class_dict
        self.pre_transforms = transforms.Compose([transforms.ToTensor()])
   
        self.post_transforms = transforms.Compose([
          transforms.Normalize(mean=[self.normalize_mean], std=[self.normalize_std])
        ])
  
    def __len__(self):
        return len(self.image_paths)
  

    def process_image(self, image_path):
        with open(image_path, 'rb') as f:
            with Image.open(f) as image:
                image = self.pre_transforms(image)
        return image

    def aug_image(self, images, center_positions):
        ## (1) 이미지들을 모두 일정한 크기로 crop을 한다. 단, 8F 4개와 32F 1개 모두 동일한 위치여야 한다.
        images = random_crop(images, self.crop_size)
        ## (2) 이미지들을 모두 일정한 각도로 rotate를 한다. (추가 AUGMENTATION)
        images = random_rotate(images)
        ## (3) 이미지에 0.3의 비율로 이미지 Cut Mix를 진행한다.
        images = cut_mix(images)
        ## (4) 이미지에 0.5의 비율로 Gaussian Random Noise를 추가한다.
        images = add_random_noise(images)
        return images

    def __getitem__(self, index):
        image_paths = self.image_paths[index]
        observed_8f_images = [self.process_image(image_paths[k]) for k in range(4)] ## 마지막은 class label index
        class_index = image_paths[-1]
        
        msr_paths = [self.get_msr_path(image_path) for image_path in image_paths]
        center_positions = [self.get_center_positions(msr_path) for msr_path in msr_paths]
        aligned_8f_images, synthesized_32f_image = self.synthesize_32f_image(observed_8f_images, center_positions)

        tensor_outputs = aligned_8f_images + [synthesized_32f_image]
        tensor_outputs = [self.post_transforms(img) for img in tensor_outputs]
        ## RANDOM CROP THE ANCHOR IMAGES
        tensor_outputs = random_crop(tensor_outputs, self.crop_size)
        ## AUGMENTATION ON THE SAMPLE IMAGES
        aug_tensor_outputs = self.aug_image(tensor_outputs, center_positions)


        return tensor_outputs + [image_paths[0]], torch.Tensor(int(image_paths[-1]), dtype = torch.int)
    
    def get_msr_path(self, image_path):
        tokens = image_path.split("/")
        msr_path = "/".join(tokens[:-1])
        folder_name = os.path.splitext(tokens[-1])[0][:-4] + "01MS.jpeg"
        acd_data_path = os.path.join(
                msr_path, ".%s" % (folder_name), "acd_data.txt"
            )
        cond_path = os.path.join(msr_path, ".%s" % (folder_name), "cond.txt")
        if os.path.exists(acd_data_path):
            return acd_data_path
        elif os.path.exists(cond_path):
            return cond_path
        else:
            index = int(folder_name[5:9])
            folder_name = folder_name.replace(str(index).zfill(4), str(index - 1).zfill(4))
            acd_data_path = os.path.join(
                    msr_path, ".%s" % (folder_name), "acd_data.txt"
                )
        cond_path = os.path.join(msr_path, ".%s" % (folder_name), "cond.txt")
        if os.path.exists(acd_data_path):
          return acd_data_path
        elif os.path.exists(cond_path):
          return cond_path

    def get_center_positions(self, msr_path):
        if "acd_data" in msr_path:
            with open(msr_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if "Center_Position_Pixel" in line:
                        head_tokens = line.split(":")
                        data_tokens = head_tokens[1].split(",")

                        position = [float(data_tokens[0]) / 10.0, float(data_tokens[1]) / 10.0]
                        return position

        elif "cond" in msr_path:
            with open(msr_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if "!Measurment_cursor_org" in line:
                        pline = " ".join(line.split())
                        tokens = pline.split(" ")
                        position = [float(tokens[1]), float(tokens[3])]
                        return position


    def save_tensor(self, file_name, tensor):
        tensor = (tensor * self.normalize_std) + self.normalize_mean
        tensor *= 255
        image = tensor.detach().cpu().clamp(0, 255).round()
        image = np.mean(image.numpy().transpose(1, 2, 0), axis=2).astype(np.uint8)
        cv2.imwrite(file_name, image)
    
    def synthesize_32f_image(self, observed_8f_images, center_positions):
        reference_shape = observed_8f_images[0].shape
        aligned_images = torch.zeros(
            (4, reference_shape[0], reference_shape[1], reference_shape[2])
        )

        def compute_aligned_image(image, offsets, bottoms):
            rfunctions = [np.ceil if bottom else np.floor for bottom in bottoms]
            image = torch.roll(image, shifts=int(rfunctions[0](offsets[0])), dims=-1)
            image = torch.roll(image, shifts=int(rfunctions[1](offsets[1])), dims=-2)
            return image

        center_positions = np.array(center_positions)
        mean_position = center_positions[0] #np.mean(center_positions, axis=0)  # goto integer
        position_shift = np.round(mean_position) - mean_position  # 2 - 2.1 = -0.1
        mean_position = mean_position + position_shift
        for k in range(4):
            center_position = center_positions[k] + position_shift
            aligned_images[k] = compute_aligned_image(
                observed_8f_images[k],
                offsets=(
                    mean_position[0] - center_position[0],
                    mean_position[1] - center_position[1],
                ),
                bottoms=(
                    mean_position[0] >= center_position[0],
                    mean_position[1] >= center_position[1],
                ),
            )
        synthesized_32f_image = aligned_images.mean(dim=0, keepdims=False)

        return [v for v in aligned_images], synthesized_32f_image

