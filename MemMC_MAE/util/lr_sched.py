import math
def adjust_learning_rate(optimizer, epoch, cfg):
    # Decay the learning rate with half-cycle cosine after warmup # 
    if epoch < cfg.warmup_epochs:
        lr = cfg.lr * epoch / cfg.warmup_epochs
    else:
        lr = cfg.min_lr + (cfg.lr - cfg.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - cfg.warmup_epochs) / (cfg.epoch - cfg.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            lr = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    
    return lr