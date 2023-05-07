import os
import torch


def load_model(cfg, model, optimizer, loss_scaler):
    if cfg.checkpoint_path:
        checkpoint = torch.load(cfg.checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            if cfg.eval == False:
                optimizer.load_state_dict(checkpoint['optimizer'])
                cfg.start_epoch = checkpoint['epoch'] + 1
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                
        

def save_model(cfg, model, optimizer, loss_scaler, epoch, str_date):
    epoch_name = str(epoch)
    output_dir = os.path.join(cfg.model_dir, str_date)
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"checkpoint-{epoch_name}.pth")
    if loss_scaler is not None:
        to_save = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "scaler": loss_scaler.state_dict(), "cfg": cfg
        }
        torch.save(to_save, checkpoint_path)
    else:
        torch.save(model.state_dict(), checkpoint_path)
    