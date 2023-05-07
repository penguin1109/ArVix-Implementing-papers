import torch
import numpy as np
import cv2
import os
from tqdm import tqdm

def get_device(model):
    params = [p for p in model.parameters() if p.requires_grad == True]
    return params[0].device

def min_max_scale(img):
    return (img - img.min()) / (img.max() - img.min())

def make_vis(input, pred, mask):
    reconstructed = np.zeros_like(input)
    reconstructed = (pred * mask) + (input * (1. - mask))
    return reconstructed

def patchify(img, cfg):
    patch_size = cfg.patch_size
    B, C, H, W = img.shape
    assert H == W
    patch_n = H // patch_size
    img = img.reshape((B, C, patch_n, patch_size, patch_n, patch_size))
    img = torch.einsum('nchpwq->nhwpqc', img)
    img = img.reshape((B, patch_n ** 2, patch_size**2*C))
    return img

def unpatchify(img, cfg):
    C = cfg.ch_in
    B, L, E = img.shape
    patch_size = cfg.patch_size
    patch_n = int(L ** 0.5)
    img = img.reshape((B, patch_n, patch_n, patch_size, patch_size, C))
    img = torch.einsum('nhwpqc->nchpwq', img)
    img = img.reshape((B, C,patch_n * patch_size, patch_n * patch_size))
    return img

def vis_mask(mask, cfg):
    B, N = mask.shape
    mask = mask.unsqueeze(-1).repeat(1, 1, cfg.patch_size ** 2 * cfg.ch_in)
    mask = unpatchify(mask, cfg)
    mask = mask.detach().cpu()
    return mask

def bgr2rgb(img):
    new_img = np.zeros_like(img)
    new_img[:, :, 0] = img[:, :, 2]
    new_img[:, :, 1] = img[:, :, 1]
    new_img[:, :, 2] = img[:, :, 0]   
    return new_img

def validate(cfg, model, test_dataloader, str_date):
    test_loop = tqdm(test_dataloader)
    device = get_device(model)
    cnt = 0
    out_path = os.path.join(cfg.result_dir, str_date)
    os.makedirs(out_path, exist_ok=True)
    model.eval()
    
    for idx, batch in enumerate(test_loop):
        img = batch.to(device)
        with torch.cuda.amp.autocast(enabled=False):
            loss, pred, mask = model(img)
        
            loss_value = loss.item()
        
            mask = vis_mask(mask, cfg)
            pred = unpatchify(pred, cfg).detach().cpu()
            img = unpatchify(img, cfg).detach().cpu()
            reconstructed = make_vis(img, pred, mask)
            reconstructed = torch.einsum('nchw->nhwc', reconstructed).numpy()
        
            reconstructed = 255. * min_max_scale(reconstructed)
            B = reconstructed.shape[0]
            for b in range(B):
                reconstructed = bgr2rgb(reconstructed[b])
                cnt += 1
                if cnt % 50 == 0:
                    cv2.imwrite(
                        os.path.join(out_path, f"{cnt}.png"), reconstructed
                    )
    model.train(True)
        
        
        
        
        
        
        
        
        