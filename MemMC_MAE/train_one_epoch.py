import torch

import sys
import math
from typing import Iterable

import util.misc as misc
import util.lr_sched as lr_sched

def train_one_epoch(
    model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, \
        device: torch.device, epoch, loss_scaler, log_writer, str_date:str, cfg
    ):
    model.train(True)
    metric_logger = misc.MetricLogger(delimeter=" ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    accum_iter = cfg.accum_iter
    
    optimizer.zero_grad()
    
    if log_writer is not None:
        # 어떤 .txt 파일에 log output을 저장하고 있는 지 출력해 준다.
        print('log_dir: {}'.format(log_writer.log_dir))
    
    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0: 
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, cfg)
        
        samples = samples.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast(): # for fast training
            loss, _, _ = model(samples, mask_ratio = cfg.mask_ratio)
        
        loss_value = loss.item()
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(), \
            update_grad=(data_iter_step+1)%accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
            
        torch.cuda.syncronize()
        
        metric_logger.update(loss=loss_value)
        
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
        # loss_value_reduce = misc.all_reduce_mean(loss_value)
        
        if log_writer is not None and (data_iter_step + 1) % accum_iter==0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
    
    return {k:meter.global_avg for k, meter in metric_logger.meters.items()}
        
        
    
    
    
