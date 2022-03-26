## BiT Hyper Rule

def get_resolution(original_resolution):
  """
  Takes (H, W) and returns (precrop, crop)
  """
  area = original_resolution[0] * original_resolution[1]
  return (160, 128) if area < 96*96 else (512, 480)

def get_schedule(dataset_size):
  if (dataset_size < 20000):
    return [100, 200, 300, 400, 500]
  elif dataset_size < 500000:
    return [500, 3000, 6000, 9000, 10000]
  else:
    return [500, 6000, 12000, 18000, 20000]

def get_lr(step, dataset_size, base_lr = 3e-2):
  """
  각각의 step 마다의 learning rate를 돌려준다
  """
  supports = get_schedule(dataset_size)
  # Linear Warming
  if step < supports[0]:
    return base_lr * step / supports[0]
  # end of training
  elif step >= supports[-1]:
    return None
  # Stir case decays by the factor of 10
  else:
    for s in supports[1:]:
      if (s < step):
        base_lr /= 10
      return base_lr