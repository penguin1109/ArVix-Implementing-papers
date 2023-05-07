import torch
from collections import defaultdict, deque
import os, cv2, time
import numpy as np
from torchvision import transforms
import datetime

class SmoothedValue(object):
    # Tracks a series of values and provides access to smoothed values over a window or the global
    # series average (즉, loss값을 몇 번의 iteration마다 시도를 한다고 볼 때 해당 iteration동안 계산되었던 값들에 평균을 내어 주는 것이다.)
    def __init__(self, window_size=20, fmt = None):
        super(SmoothedValue, self).__init__()
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size) # window_size 보다 많은 수가 들어가면 자동으로 제일 이전에 append된 값을 빼주게 된다.
        self.window_size = window_size
        self.fmt = fmt
        self.count = 0
        self.total = 0.0
    
    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n
    
    def synchronize_between_processes(self):
        # 만약에 여러개의 process를 병렬적으로 처리한다고 할 때에만 필요하다.
        pass
    
    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deuque), dtype=torch.float32)
        return d.mean().item()
    
    @property
    def global_avg(self):
        return self.total / self.count
    
    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )
        
class MetricLogger(object):
    def __init__(self, delimeter="\t"):
        super(MetricLogger, self).__init__()
        self.meters = defaultdict(SmoothedValue)
        self.delimeter = delimeter
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)
        
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{} : {}".format(name, str(meter)))
        return self.delimeter.join(loss_str)

    def add_meter(self, name, value):
        self.meters[name] = value
        
    def log_every(self, iterable, print_freq, header):
        i = 0
        if not header:header=''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt = '{avg:.4f}')
        data_time = SmoothedValue(fmt = '{avg:.4f}')
        space_fmt = ":" + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header, '[{0' + space_fmt + '}/{1}]', 
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimeter.join(log_msg)
        MB = 1024.0 ** 2
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            # 여기서 print_freq의 주기성을 갖고 현재 lr, loss,등의 값들을 print하도록 하는 것임.
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds = int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string, meters=str(self),
                        time = str(iter_time), data = str(data_time),
                        memory = torch.cuda.max_memory_allocated() / MB
                    ))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string, meters = str(self),
                        time = str(iter_time), data = str(data_time)
                    ))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds = int(total_time)))
        print('{} Total Time: {} ({:.4f}s / it)'.format(
            header, total_time_str, total_time / len(iterable)
        ))

def get_grad_norm(parameters, norm_type = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type= float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].device
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        # torch.norm returns the matrix norm or vector norm of a given tensor.
        total_norm = torch.norm(torch.stack([
            torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters
        ]), norm_type)
        
    return total_norm

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"
    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()
    
    def __call__(self, loss, optimizer, parameters=None, clip_grad=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            """ What does the .unscale_() method do?
            - Divides the optimizer's gradient tensors by the scale factor.
            """
            if clip_grad is not None:
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm(parameters)
            """ What does the .step() method do?
            1. Internally invokes unscale_(optimizer). Gradients are checked for infs/NaNs
            2. If no Inf/NaN gradients are found, invokes optimizer.stpe() using the unscaled gradients. Otherwise, optimizer.step()
            is skipped to avoid corrupting the params.
            """
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm=None
        return norm
    
    def state_dict(self):
        # returns the state of the scaler as a dict
        return self._scaler.state_dict()
    
    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)



def patchify(img, patch_size):
    B, C, H, W = img.shape
    n = H // patch_size
    img = img.reshape((B, C, n, patch_size, n, patch_size))
    img = torch.einsum('nchpwq->nhwpqc', img)
    img = img.reshape((B, n * n, patch_size * patch_size * C))
    return img

def unpatchify(img, patch_size, channel_size):
    B, L, E = img.shape
    n = int(L ** (0.5)) # 한변에의 patch의 개수
    hp = wp = int((E // channel_size) ** 0.5)
    C = channel_size
    img = img.reshape((B, n, n, hp, wp, C))
    img = torch.einsum('nhwpqc->nchpwq', img)
    img = img.reshape((B, C, hp * n, wp * n))
    return img

def visualize_img(img, pred, mask, cfg):
    pred = unpatchify(pred, cfg.patch_size, cfg.ch_in) # [B, C, H, W]
    pred = pred.detach().cpu()#.numpy()
    img = img.detach().cpu()# .numpy()

    B, N = mask.shape # [Batch Size, #of patches]
    # [Batch Size, #of patches, 1] -> [Batch Size, #patches, patch size]
    mask = mask.unsqueeze(-1).repeat(1, 1, cfg.patch_size**2 *cfg.ch_in)
    mask = unpatchify(mask,cfg.patch_size, cfg.ch_in)
    mask = mask.detach().cpu() # .numpy()

    vis = (pred * mask) + (img * (1. - mask))
    vis = torch.einsum('nchw->nhwc', vis).numpy() # [B, H, W, C]

    vis = (vis - vis.min()) / (vis.max() - vis.min())
    vis *= 255. # 시각화 할 수 있게 변경
    return vis[0]

def bgr2rgb(img):
    H, W, C = img.shape
    new_img = np.zeros_like(img)
    new_img[:, :, 0] = img[:, :, 2]
    new_img[:, :, 1] = img[:, :, 1]
    new_img[:, :, 2] = img[:, :, 0]

    return new_img
  
def img2tensor(img_path, cfg):
    img = cv2.imread(img_path)
    aug = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(cfg.img_size),
        transforms.Normalize(cfg.mean, cfg.std)
    ])
    img = aug(img)
    img = torch.unsqueeze(img, 0)
    return img

def save_prediction(cfg, model, img_path, device):
    # device = model.device
    model.eval()

    from tqdm import tqdm
    from datetime import date
    today = date.today()
    today = today.strftime("%m_%d_%y")
    if cfg.prefix != '':
        today = f"{today}_{cfg.prefix}"

    out_dir = os.path.join(cfg.result_dir, today)
    os.makedirs(out_dir, exist_ok=True)


    img = img2tensor(img_path).to(device)

    with torch.cuda.amp.autocast(enabled=False):
        loss, pred, mask = model(img)
        vis = visualize_img(img, pred, mask, cfg) # [H, W, C]

        print("SAVING VISUALIZED IMAGE")
        cv2.imwrite(
             os.path.join(out_dir, f"{idx}.png"), vis
        )
    model.train(True)
