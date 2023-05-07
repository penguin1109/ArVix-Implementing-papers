class CFG(object):
  def __init__(self):
    self.eval = False # True 
    self.tensorboard_dir = None
    self.logging_dir = '/content/drive/MyDrive/Competition/dacon/anomal_classify/log'
    self.result_dir = '/content/drive/MyDrive/Competition/dacon/anomal_classify/result' # predicted image 저장 경로
    self.model_dir = '/content/drive/MyDrive/Competition/dacon/anomal_classify/model' # trained model 저장 경로 
    self.prefix = 'my_mae'

    ## MODEL CONFIGURATIONS ##
    self.ch_in = 3
    self.img_size = 224
    self.patch_size = 16
    self.norm_pix_loss = False
    self.encoder_embed_dim = 768 # 512
    self.encoder_layers = 12
    self.encoder_num_heads = 12
    self.encoder_memory = False
    self.decoder_embed_dim = 512 # 768 
    # the MAE uses an asymmetric encoder-decoder structure. But it is called 'asymmetric' by that the number of layers are different (the decoder is more lightweight)
    self.decoder_layers =  8 # 12 # 8
    self.decoder_num_heads = 16 # 8

    ## DATA CONFIGURATIONS ##
    self.mean =  [0.485, 0.456, 0.406] # [0.0, 0.0, 0.0] # [0.5,0.5,0.5] # [0.57677052, 0.59262288, 0.60103319] 
    self.std =  [0.229, 0.224, 0.225] # [1.0, 1.0, 1.0] # [0.5, 0.5,0.5] # [0.09492197, 0.09107896, 0.08727206]

    ## TRAIN CONFIGURATIONS ##
    self.mask_ratio = 0.75
    self.optim = 'adamw'
    self.base_lr=1e-3
    self.lr = None  # 1e-3 # 1.5e-4 # base learning rate
    self.min_lr=0
    self.accum_iter = 1 # accumulate gradient iterations
    self.warmup_epochs = 40
    self.norm_pix_loss = True # False
    self.batch_size = 64 # 16 # 작을수 밖에 없는 batch size도 상당히 영향을 미치는게 아닐까?
    self.start_epoch = 0 # training을 이어서 하기 위해서는 start epoch의 정보도 필요하다.
    self.epoch = 400
    self.weight_decay = 0.05
    self.accum_iter = 1 # 몇 iteration마다 learning rate를 조정할 것인지 
    self.eval_epoch = 10 # 5epoch만큼 학습 시켰을 때 test_split으로 evaluation을 해야 한다.
    # -> 근데 우선은 evaluation을 한다는게 classification 전에 이미지를 제대로 생성해 내는지를 확인해 보고 싶은 것이다.

