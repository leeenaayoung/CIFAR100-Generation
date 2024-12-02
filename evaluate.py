import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as P
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.linalg import sqrtm
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader

superclass_mapping = {
    0: [4, 30, 55, 72, 95],  # aquatic mammals
    1: [1, 32, 67, 73, 91],  # fish
    2: [54, 62, 70, 82, 92],  # flowers
    3: [9, 10, 16, 28, 61],  # food containers
    4: [0, 51, 53, 57, 83],  # fruit and vegetables
    5: [22, 39, 40, 86, 87],  # household electrical devices
    6: [5, 20, 25, 84, 94],  # household furniture
    7: [6, 7, 14, 18, 24],  # insects
    8: [3, 42, 43, 88, 97],  # large carnivores
    9: [12, 17, 37, 68, 76],  # large man-made outdoor things
    10: [23, 33, 49, 60, 71],  # large natural outdoor scenes
    11: [15, 19, 21, 31, 38],  # large omnivores and herbivores
    12: [34, 63, 64, 66, 75],  # medium-sized mammals
    13: [26, 45, 77, 79, 99],  # non-insect invertebrates
    14: [2, 11, 35, 46, 98],  # people
    15: [27, 29, 44, 78, 93],  # reptiles
    16: [36, 50, 65, 74, 80],  # small mammals
    17: [47, 52, 56, 59, 96],  # trees
    18: [8, 13, 48, 58, 90],  # vehicles 1
    19: [41, 69, 81, 85, 89]   # vehicles 2
}


def load_inception_net():
  inception_model = inception_v3(pretrained=True, transform_input=False)
  inception_model = WrapInception(inception_model.eval()).cuda()
  return inception_model


class WrapInception(nn.Module):
  def __init__(self, net):
    super(WrapInception,self).__init__()
    self.net = net
    self.mean = P(torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1),
                  requires_grad=False)
    self.std = P(torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1),
                 requires_grad=False)
  def forward(self, x):
    # Normalize x
    x = (x + 1.) / 2.0
    x = (x - self.mean) / self.std
    # Upsample if necessary
    if x.shape[2] != 299 or x.shape[3] != 299:
      x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
    # 299 x 299 x 3
    x = self.net.Conv2d_1a_3x3(x)
    # 149 x 149 x 32
    x = self.net.Conv2d_2a_3x3(x)
    # 147 x 147 x 32
    x = self.net.Conv2d_2b_3x3(x)
    # 147 x 147 x 64
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    # 73 x 73 x 64
    x = self.net.Conv2d_3b_1x1(x)
    # 73 x 73 x 80
    x = self.net.Conv2d_4a_3x3(x)
    # 71 x 71 x 192
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    # 35 x 35 x 192
    x = self.net.Mixed_5b(x)
    # 35 x 35 x 256
    x = self.net.Mixed_5c(x)
    # 35 x 35 x 288
    x = self.net.Mixed_5d(x)
    # 35 x 35 x 288
    x = self.net.Mixed_6a(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6b(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6c(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6d(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6e(x)
    # 17 x 17 x 768
    # 17 x 17 x 768
    x = self.net.Mixed_7a(x)
    # 8 x 8 x 1280
    x = self.net.Mixed_7b(x)
    # 8 x 8 x 2048
    x = self.net.Mixed_7c(x)
    # 8 x 8 x 2048
    pool = torch.mean(x.view(x.size(0), x.size(1), -1), 2)
    # 1 x 1 x 2048
    logits = self.net.fc(F.dropout(pool, training=False).view(pool.size(0), -1))
    # 1000 (num_classes)
    return pool, logits
  

def get_net_output(train_loader, net,device):
  pool, logits, labels = [], [], []

  for i, (x, y) in enumerate(train_loader):
      x = x.to(device)
      with torch.no_grad():
        pool_val, logits_val = net(x)
        pool += [np.asarray(pool_val.cpu())]
        logits += [np.asarray(F.softmax(logits_val, 1).cpu())]
        labels += [np.asarray(y.cpu())]
  pool, logits, labels = [np.concatenate(item, 0) for item in [pool, logits, labels]]
  return pool, logits, labels




def accumulate_inception_activations(sample, net, num_inception_images=50000):
  pool, logits, labels = [], [], []
  i = 0
  while (torch.cat(logits, 0).shape[0] if len(logits) else 0) < num_inception_images:
    with torch.no_grad():
      images, labels_val = sample()
      pool_val, logits_val = net(images.float())
      pool += [pool_val]
      logits += [F.softmax(logits_val, 1)]
      labels += [labels_val]
  return torch.cat(pool, 0), torch.cat(logits, 0), torch.cat(labels, 0)

  
def calculate_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid
  
def calculate_inception_score(pred, num_splits=10):
  scores = []
  for index in range(num_splits):
    pred_chunk = pred[index * (pred.shape[0] // num_splits): (index + 1) * (pred.shape[0] // num_splits), :]
    kl_inception = pred_chunk * (np.log(pred_chunk) - np.log(np.expand_dims(np.mean(pred_chunk, 0), 0)))
    kl_inception = np.mean(np.sum(kl_inception, 1))
    scores.append(np.exp(kl_inception))
  return np.mean(scores), np.std(scores)



def calculate_intra_fid(pool, logits,labels ,g_pool, g_logits, g_labels, chage_superclass=True):
  intra_fids = []
  super_class = super_class_mapping()
  
  super_labels = [super_class[i] for i in labels]
  super_labels = np.array(super_labels)
  
  if chage_superclass:
    g_super_labels = [super_class[i] for i in g_labels]
    g_super_labels = np.array(g_super_labels)
  else:
    g_super_labels = np.array(g_labels.cpu())
  
  for super_idx, _ in superclass_mapping.items():
    mask = (super_labels == super_idx)
    g_mask = (g_super_labels == super_idx)
    
    pool_low = pool[mask]
    g_pool_low = g_pool[g_mask]
    
    assert 2500 == len(g_pool_low), "super-classes count error"
    if len(pool_low) == 0 or len(g_pool_low) == 0:
      continue
    
    mu, sigma = np.mean(g_pool_low.cpu().numpy(), axis=0), np.cov(g_pool_low.cpu().numpy(), rowvar=False)
    mu_data, sigma_data = np.mean(pool_low, axis=0), np.cov(pool_low, rowvar=False)
    
    fid = calculate_fid(mu, sigma, mu_data, sigma_data)
    intra_fids.append(fid)
    
  return np.mean(intra_fids), intra_fids
    
  
def super_class_mapping():
  class_to_superclass = [None] * 100
  for super_idx, class_indices in superclass_mapping.items():
    for class_idx in class_indices:
      class_to_superclass[class_idx] = super_idx
  return class_to_superclass


def evaluate_model(sample):
    net = load_inception_net()

    norm_mean = [0.5,0.5,0.5]
    norm_std = [0.5,0.5,0.5]
    image_size = 32,32

    train_transform = []
    train_transform = transforms.Compose(train_transform + [
                     transforms.ToTensor(),
                        transforms.Resize((299, 299)),
                     transforms.Normalize(norm_mean, norm_std)])


    train_dataset = torchvision.datasets.CIFAR100(
        root="./data",  # 데이터 저장 경로
        train=True,     # 학습용 데이터셋
        download=True,  # 데이터셋 다운로드
        transform=train_transform
    )
    
    # DataLoader 배치 사이즈 줄이기
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 메모리 절약을 위한 설정
    torch.cuda.empty_cache()  # 시작 전 캐시 비우기
    
    with torch.cuda.amp.autocast():  # mixed precision 사용
        pool, logits, labels = get_net_output(device="cuda:0", train_loader=train_loader, net=net)
        mu_data, sigma_data = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
        
        g_pool, g_logits, g_labels = accumulate_inception_activations(sample, net, 50000)
        
        g_pool = g_pool[:50000]
        g_logits = g_logits[:50000]
        g_labels = g_labels[:50000]
        
        mu = np.mean(g_pool.cpu().numpy(), axis=0)
        sigma = np.cov(g_pool.cpu().numpy(), rowvar=False)
        
    # 메트릭 계산
    fid = calculate_fid(mu_data, sigma_data, mu, sigma)
    m, conv = calculate_inception_score(g_logits.cpu().numpy(), 10)
    intra_fids_mean, intra_fids = calculate_intra_fid(pool, logits, labels, g_pool, g_logits, g_labels, chage_superclass=False)
    
    metrics = {
        'inception_score': {'mean': m, 'std': conv},
        'fid': fid,
        'intra_fid': {'mean': intra_fids_mean, 'superclass_fids': intra_fids}
    }
    
    # 메모리 정리
    torch.cuda.empty_cache()
    
    return metrics
    