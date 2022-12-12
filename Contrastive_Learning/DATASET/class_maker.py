""" Step (1) GETTING THE UNIQUE CLASSES
<class_oper>
- 조합 1: 그냥 RECIPE NAME만 (하지만 동일한 recipe이름을 사용해서 supervised learning을 하면 무늬 그대로를 학습할 확률이 높다.)
- 조합 2: LOT_CD + OPER_ID + TECH_CD
- 조합 3: LOT_CD + TECH_CD
- 조합 4: OPER_ID + TECH_CD
- 조합 5: RECIPE NAME + TECH_CD
"""
import torch
import torch.nn as nn
import os, glob
import torch.nn.functional as F
from torchvision import transforms
import torch.utils.data as data
import numpy as np
from gli_aims.datasets.aims_assets import AIMSAssets

OPER_DICT={
    "1": ['recipe'],
    "2": ['lot', 'oper', 'tech'],
    "3": ['lot', 'tech'],
    "4": ['oper', 'tech'],
    "5": ['recipe', 'tech']
}
class SupervisedClassMaker(object):
  def __init__(self, 
               asset_configuration,
               data_configurations, 
               parameter_mappings,
               meta_datas,
               class_oper=2):
    super().__init__()
    """ Making the available classes for the supervised contrastive learning to train the degradation encoder
    - 각각의 class에 맞는 image_dir, condition에 대한 정보가 들어 있을 것임
    - RECIPE_NAME, OPER_ID, LOT_CD, TECH_CD
    """ 
    self.aims_assets = AIMSAssets(asset_configuration)
    self.parameter_mappings = parameter_mappings
    self.meta_datas = meta_datas
    self.class_mode = OPER_DICT[str(class_oper)]
    self.class_dict = []
    self.image_paths = []
    self.folder_paths = []
    self.inline_paths = []
  


    for data_cfg, param_map, meta_data in zip(
        data_configurations, parameter_mappings, meta_datas
      ):
      recipe_name = data_cfg[0]['cd_recipe_name']
      target_recipe, target_data_structure = self.aims_assets.recipes[recipe_name].instance
      source_indices = []
      for i in range(len(param_map)):
        source_indices_ = []
        source_config = param_map[i]
        for source_param in source_config["source-parameters"]:
          source_indices_.append(
              target_data_structure.parameter_sets[source_param].image_index
          )
        source_indices_ = np.array(source_indices_).transpose().tolist()
        source_indices += source_indices_


        for data_cfg_ in data_cfg:
          index_to_image_map = {}
          image_names = sorted(glob.glob(
              os.path.join(data_cfg_["folder_path"], "*.tif")
          ))
          ## image name에서 직접 구할 수 있는 image index와 target structure data에서 알 수 있는 index는 동일하다.
          for image_name in image_names:
            index_to_image_map[int(os.path.basename(image_name)[5:9])] = image_name
          class_label = self.make_unique_class(recipe_name)
          self.class_dict.append(class_label)
          for source in source_indices_:
            self.image_paths.append(
                [index_to_image_map[index] for index in source] + \
                      [class_label] ## 8F이미지 4개의 파일 경로가 모두 저장이 되어 있음 
            )
          self.folder_paths.append(data_cfg_["folder_path"])
          self.inline_paths.append(data_cfg_["inline_path"])

    self.class_dict = list(set(self.class_dict))
    self.class_dict = {key: value for (key, value) in zip(self.class_dict, [int(i) for i in range(len(self.class_dict))])}
    def change_first(arr):
      cls = self.class_dict[arr[-1]]
      arr[-1] = cls
      return arr

    self.image_paths.apply(change_first)
 

  def make_unique_class(self, recipe_name, meta_data):
    class_label = []
    if 'recipe' in self.class_mode:
      class_label.append(recipe_name)
    if 'lot' in self.class_mode:
      class_label.append(meta_data['LOT'])
    if 'oper' in self.class_mode:
      class_label.append(meta_data['OPER'])
    if 'tech' in self.class_mode:
      class_label.append(meta_data['TECH'])

    return ' '.join(class_label)

      
      

  

