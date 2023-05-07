import torch
import torch.distributed as dist
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import timm.optim.optim_factory as optim_factory

from mae.mae_model import AnnoMAE
from util.dataset import MVTecDataset
from config.mae import CFG
from util.misc import NativeScalerWithGradNormCount as NativeScaler
# from util.misc import save_prediction
from validate import validate

from util.tools import load_model, save_model
from train_one_epoch import train_one_epoch
from validate import validate



import time
import numpy as np
from pytz import timezone
import os
import datetime
import json

def run(cfg):
    print(f"CUDA AVAILABLE: {torch.cuda.is_available()}")
    today = datetime.datetime.now(timezone("Asia/Seoul"))
    str_date = today.strftime("%m_%d_%Y")
    device = torch.device('cuda')
    if cfg.tensorboard_dir is not None:
        log_writer = SummaryWriter(cfg.tensorboard_dir)
    else:
        log_writer = None
    loss_scaler = NativeScaler()
    
    # fix the seed for reproducibility
    seed = 0 + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    cudnn.benchmark = True
    
    # set up the dataset
    train_dataset = MVTecDataset(cfg)
    test_dataset = MVTecDataset(cfg, mode='test')
    sampler_train = torch.utils.data.RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=sampler_train, batch_size=cfg.batch_siez, pin_memory=True, drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, num_workers=1, drop_last=False
    )
    
    # set up the model
    model = AnnoMAE(
        cfg.img_size, cfg.patch_size, cfg.ch_in, cfg.encoder_embed_dim, cfg.encoder_layers, cfg.encoder_num_heads,
        cfg.decoder_embed_dim, cfg.decoder_layers, cfg.decoder_num_heads
    )
    model = model.to(device)
    
    load_model(cfg, model, optimizer, loss_scaler)
    
    # set up the optimizer
    lr = cfg.base_lr * cfg.batch_size / 256 if cfg.lr is None else cfg.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr, 
                                  betas=(0.9, 0.95), weight_decay=cfg.weight_decay)

    
    # start training
    print("START TRAINING...")
    start_time = time.time()
    for epoch in range(cfg.start_epoch, cfg.epoch):
        train_stats = train_one_epoch(
            model, train_dataloader, optimizer, device, epoch, loss_scaler, log_writer, str_date, cfg
        )
        
    
        if epoch % 20 == 0 or epoch + 1 == cfg.epoch:
            save_model(
                cfg, model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, str_date=str_date
            )
            # save_prediction(cfg, model, test_dataloader, device)
            validate(cfg, model, test_dataloader, str_date)
        
        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, 'epoch': epoch}
    
        if cfg.logging_dir is not None:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(cfg.logging_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")    
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds = int(total_time)))
    print(f'TRAINING TIME : {total_time_str}')
    
if __name__ == "__main__":
    cfg = CFG()
    if cfg.logging_dir:
        os.makedirs(cfg.logging_dir, exist_ok=True)
    run(cfg)