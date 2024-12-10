import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import numpy as np
import os
import json
import random
import math
from torch.autograd import grad as torch_grad

import matplotlib.pyplot as plt
import wandb
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms

from stylegan2_model import StyleGAN2Generator, StyleGAN2Discriminator
from data_loader import get_dataloader
from evaluate import evaluate_model
from diff_aug import apply_diffaug

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # GPU 여러 개 사용할 경우 사용
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # 완벽한 재현성을 위해선 False로 하는 게 맞지만 성능의 저하를 일으킬 수 있음
    np.random.seed(seed)
    random.seed(seed)
set_seed(0)

def gen_hinge_loss(fake):
    return fake.mean()

def hinge_loss(fake, real):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()

def image_noise(n, im_size, device):
    return torch.empty(n, 1, im_size, im_size, device=device).uniform_(0., 1.)

def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

class StyleVectorizer(nn.Module):
    def __init__(self, emb=512, depth=8, lr_mul = 0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)

def calc_pl_lengths(styles, images):
    device = images.device
    num_pixels = images.shape[2] * images.shape[3]
    pl_noise = torch.randn(images.shape, device=device) / math.sqrt(num_pixels)
    outputs = (images * pl_noise).sum()

    pl_grads = torch_grad(outputs=outputs, inputs=styles,
                          grad_outputs=torch.ones(outputs.shape, device=device),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]

    return (pl_grads ** 2).sum(dim=1).mean().sqrt()

# CIFAR-100 클래스명
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

SUPERCLASS_NAMES = [
   'aquatic_mammals', 'fish', 'flowers', 'food_containers', 
   'fruit_and_vegetables', 'household_electrical_devices', 
   'household_furniture', 'insects', 'large_carnivores',
   'large_man-made_outdoor_things', 'large_natural_outdoor_scenes',
   'large_omnivores_and_herbivores', 'medium_mammals',
   'non-insect_invertebrates', 'people', 'reptiles',
   'small_mammals', 'trees', 'vehicles_1', 'vehicles_2'
]

CLASS_TO_SUPERCLASS = [4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11, 6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5, 5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17, 10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13, 16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13]

intra_fids_list = {
    4: 0, 30: 0, 55: 0, 72: 0, 95: 0,       # aquatic mammals
    1: 1, 32: 1, 67: 1, 73: 1, 91: 1,       # fish
    54: 2, 62: 2, 70: 2, 82: 2, 92: 2,      # flowers
    9: 3, 10: 3, 16: 3, 28: 3, 61: 3,       # food containers
    0: 4, 51: 4, 53: 4, 57: 4, 83: 4,       # fruit and vegetables
    22: 5, 39: 5, 40: 5, 86: 5, 87: 5,      # household electrical devices
    5: 6, 20: 6, 25: 6, 84: 6, 94: 6,       # household furniture
    6: 7, 7: 7, 14: 7, 18: 7, 24: 7,        # insects
    3: 8, 42: 8, 43: 8, 88: 8, 97: 8,       # large carnivores
    12: 9, 17: 9, 37: 9, 68: 9, 76: 9,      # large man-made outdoor things
    23: 10, 33: 10, 49: 10, 60: 10, 71: 10, # large natural outdoor scenes
    15: 11, 19: 11, 21: 11, 31: 11, 38: 11, # large omnivores and herbivores
    34: 12, 63: 12, 64: 12, 66: 12, 75: 12, # medium-sized mammals
    26: 13, 45: 13, 77: 13, 79: 13, 99: 13, # non-insect invertebrates
    2: 14, 11: 14, 35: 14, 46: 14, 98: 14,  # people
    27: 15, 29: 15, 44: 15, 78: 15, 93: 15, # reptiles
    36: 16, 50: 16, 65: 16, 74: 16, 80: 16, # small mammals
    47: 17, 52: 17, 56: 17, 59: 17, 96: 17, # trees
    8: 18, 13: 18, 48: 18, 58: 18, 90: 18,  # vehicles 1
    41: 19, 69: 19, 81: 19, 85: 19, 89: 19  # vehicles 2
}


def gradient_penalty(discriminator, real_samples, fake_samples, device, labels):
    """Calculate WGAN-GP gradient penalty."""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates, _ = discriminator(interpolates, labels)  # 첫 번째 반환값만 사용
    
    gradients = grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

class MetricsLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.metrics = {
            'inception_score': {'mean': [], 'std': []},
            'fid': [],
            'intra_fid': {'mean': [], 'superclass_fids':[]},
            'd_loss': [],
            'g_loss': []
        }
        
        self.plots_dir = os.path.join(log_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def update(self, epoch, metrics_dict):
        """메트릭 업데이트 및 저장"""
        def convert_to_python_type(obj):
            if isinstance(obj, np.int32) or isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return convert_to_python_type(obj.tolist())
            elif isinstance(obj, list):
                return [convert_to_python_type(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_python_type(value) for key, value in obj.items()}
            else:
                return obj
    
        # 메트릭 결과를 Python 네이티브 타입으로 변환
        metrics_dict = convert_to_python_type(metrics_dict)
        
        log_dict = {
            'epoch': epoch,
            'metrics/d_loss': metrics_dict.get('d_loss', 0),
            'metrics/g_loss': metrics_dict.get('g_loss', 0)
        }
        
        for key, value in metrics_dict.items():
            if key in ['d_loss', 'g_loss', 'fid']:
                self.metrics[key].append(value)
            elif key == 'inception_score':
                self.metrics[key]['mean'].append(value['mean'])
                self.metrics[key]['std'].append(value['std'])
            elif key == 'intra_fid':
                self.metrics[key]['mean'].append(value['mean'])
                if 'std' in value:  # std가 있는 경우에만 추가
                    self.metrics[key]['std'].append(value['std'])
                self.metrics[key]['superclass_fids'].append(value['superclass_fids'])
        
        if 'inception_score' in metrics_dict:
            log_dict.update({
                'metrics/inception_score/mean': metrics_dict['inception_score']['mean'],
                'metrics/inception_score/std': metrics_dict['inception_score']['std'],
                'metrics/fid': metrics_dict['fid'],
                'metrics/intra_fid/mean': metrics_dict['intra_fid']['mean'],
            })
            
            if 'std' in metrics_dict['intra_fid']:
                log_dict['metrics/intra_fid/std'] = metrics_dict['intra_fid']['std']
            
            # Superclass Intra-FID Plot
            if 'superclass_fids' in metrics_dict['intra_fid']:
                fig = plt.figure(figsize=(15, 8))
                bars = plt.bar(SUPERCLASS_NAMES, metrics_dict['intra_fid']['superclass_fids'])
                plt.xticks(rotation=45, ha='right')
                plt.title('Superclass Intra-FID Scores')
                plt.xlabel('Superclass')
                plt.ylabel('Intra-FID Score')
                
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}',
                            ha='center', va='bottom', rotation=0)
                
                plt.tight_layout()
                
                # 로컬에 저장
                save_path = os.path.join(self.plots_dir, f'superclass_intra_fid_epoch_{epoch}.png')
                plt.savefig(save_path)
                
                # wandb에 로깅
                log_dict["plots/superclass_intra_fid"] = wandb.Image(fig)
                plt.close()
        
        wandb.log(log_dict)
        
        # JSON으로 저장하기 전에 모든 값이 직렬화 가능한지 확인
        with open(os.path.join(self.log_dir, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=4)
           

def wrap_text(text, max_width, font, draw):
    """
    텍스트를 주어진 너비(max_width)에 맞게 줄바꿈
    """
    lines = []
    words = text.split()  
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        text_width = draw.textlength(test_line, font=font)
        if text_width <= max_width:
            current_line = test_line
        else:
            if current_line:  
                lines.append(current_line)
            current_line = word  

    if current_line:
        lines.append(current_line)

    # 긴 단어 처리 (단어 자체가 max_width를 초과하는 경우)
    wrapped_lines = []
    for line in lines:
        while draw.textlength(line, font=font) > max_width:
            split_index = len(line) // 2  
            wrapped_lines.append(line[:split_index])
            line = line[split_index:]
        wrapped_lines.append(line)  

    return wrapped_lines


def save_generated_images(generator, style_vectorizer,epoch, save_dir, superclass_names, intra_fids_list, n_samples=9, device='cuda'):
    """
    각 슈퍼클래스별로 3x3 이미지를 생성하고, 이를 4x5 그리드로 배치하여 저장
    """
    generator.eval()
    with torch.no_grad():
        # 슈퍼클래스별로 n_samples씩 생성
        n_superclasses = len(superclass_names)
        total_samples = n_superclasses * n_samples  # 총 샘플 수 계산
        batch_size = total_samples
        
        z = torch.randn(batch_size, generator.latent_dim, device=device)

        # Generate fake images
        w_styles = style_vectorizer(z)
        spatial_noise = image_noise(batch_size, 32, device)  # Spatial noise
        fake_labels = []

        for superclass_idx in range(n_superclasses):
            subclass_indices = [i for i, x in enumerate(CLASS_TO_SUPERCLASS) if x == superclass_idx]
            fake_labels.extend(np.random.choice(subclass_indices, n_samples))  # n_samples만큼 생성
        # fake_labels를 Tensor로 변환
        fake_labels = torch.tensor(fake_labels, device=device)  # (total_samples,) 크기

        assert fake_labels.shape[0] == total_samples, "fake_labels 크기가 z와 일치하지 않습니다."

        fake_labels = torch.tensor(
                [intra_fids_list[label.item()] for label in fake_labels],
                device=device
            )
        # Generator에 입력
        fake_images = generator(w_styles, fake_labels, spatial_noise).cpu().detach() 

    # 생성된 이미지 값 범위 확인
    print(f"Fake images min: {fake_images.min()}, max: {fake_images.max()}")
    
    # 캔버스 설정
    ncols = 5  # 한 행에 표시할 슈퍼클래스 수
    n_superclass_rows = (n_superclasses + ncols - 1) // ncols
    img_size = 32  # 이미지 크기
    padding = 6  # 이미지 간 패딩
    label_padding = 10  # 라벨 간 간격
    horizontal_padding = 20  # 양옆 공백 추가
    vertical_padding = 20  # 위아래 공백 추가
    font = ImageFont.load_default()

    # Create a temporary image and drawing object for text measurements
    temp_img = Image.new("RGB", (1, 1), "white")
    temp_draw = ImageDraw.Draw(temp_img)

    # 라벨 높이 계산
    label_heights = []
    for superclass_name in superclass_names:
        lines = wrap_text(superclass_name, img_size * 3, font, temp_draw)
        bbox = temp_draw.textbbox((0, 0), lines[0], font=font)
        line_height = bbox[3] - bbox[1]
        label_heights.append(len(lines) * line_height)

    # 전체 캔버스 크기 계산
    max_label_height = max(label_heights) + label_padding
    subclass_grid_size = img_size * 3 + padding * 2
    total_width = (
        ncols * subclass_grid_size + (ncols - 1) * padding + horizontal_padding * 2
    )
    total_height = (
        n_superclass_rows * (max_label_height + subclass_grid_size) +
        (n_superclass_rows - 1) * padding +
        vertical_padding * 2
    )

    # 캔버스 생성
    canvas = Image.new('RGB', (total_width, total_height), 'white')
    draw = ImageDraw.Draw(canvas)

    # 이미지와 라벨 배치
    for idx, superclass_name in enumerate(superclass_names):
        row, col = divmod(idx, ncols)
        start_x = col * (subclass_grid_size + padding) + horizontal_padding  
        start_y = row * (max_label_height + subclass_grid_size + padding) + vertical_padding  

        # 라벨 출력
        lines = wrap_text(superclass_name, img_size * 3, font, draw)
        text_y = start_y
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = start_x + (subclass_grid_size - text_width) // 2
            draw.text((text_x, text_y), line, fill="black", font=font)
            text_y += text_height

        # 3x3 이미지 그리드 출력
        subclass_start_idx = idx * n_samples
        for i in range(3):
            for j in range(3):
                img_idx = subclass_start_idx + i * 3 + j
                if img_idx >= total_samples:
                    break
                img = (fake_images[img_idx] + 1) / 2
                img = transforms.ToPILImage()(img.clamp(0, 1))
                img = img.resize((img_size, img_size))
                img_x = start_x + j * (img_size + padding)
                img_y = start_y + max_label_height + i * (img_size + padding)
                canvas.paste(img, (img_x, img_y))

    # 이미지 저장
    save_path = os.path.join(save_dir, f'samples_epoch_{epoch}.png')
    canvas.save(save_path)
    wandb.log({
        "Generated Samples": wandb.Image(canvas),
        "epoch": epoch
    })

    print(f"Saved generated images to {save_path}")
    
def evaluate_epoch(generator, style_vectorizer, real_dataloader, n_eval_samples=50000, batch_size=32, device='cuda'):
    """모델 평가 함수"""
    generator.eval()
    def sample_generator():
        # 작은 배치 사이즈로 생성
        labels = []
        samples_per_class = batch_size // 20  # 각 클래스당 샘플 수
        
        for i in range(20):
            labels.extend([i] * samples_per_class)
            
        # tensor로 변환하고 셔플
        labels = torch.tensor(labels, device=device)
        perm = torch.randperm(batch_size)
        labels = labels[perm]
        
        # 이미지 생성
        z = torch.randn(batch_size, generator.latent_dim, device=device)
        spatial_noise = torch.randn(batch_size, 1, 32, 32, device=device)

        # Generate fake images
        w_styles = style_vectorizer(z)
        spatial_noise = image_noise(batch_size, 32, device)  # Spatial noise
        
        # 메모리 절약을 위해 gradient 계산 비활성화
        with torch.cuda.amp.autocast():  # mixed precision 사용
            with torch.no_grad():
                images = generator(w_styles, labels, spatial_noise)
                
        return images, labels

    try:
        # 캐시 비우기
        torch.cuda.empty_cache()
        
        # evaluate_model 함수를 사용해 모든 메트릭 계산
        metrics = evaluate_model(sample_generator)
        
    finally:
        # 평가 후 메모리 정리
        torch.cuda.empty_cache()
    
    return metrics
    
    
def train_stylegan2( 
    n_epochs=100,
    batch_size=64,
    latent_dim=512,
    device='cuda:0',
    checkpoint_dir='savepoints',
    checkpoint_freq=10,
    eval_freq=5,
    resume=None,
    log_freq=100,
    lr=0.0002,
    num_workers=4,
    n_eval_samples=2500,
    eval_batch_size=32,
    diffaug_policy="translation,cutout"  # DiffAug 정책 추가
):
    best_metrics = [0,0,0,100]
    """StyleGAN2 훈련 함수"""
    global intra_fids_list 
    # 모델 초기화
    generator = StyleGAN2Generator(latent_dim).to(device)
    discriminator = StyleGAN2Discriminator().to(device)
    style_vectorizer = StyleVectorizer().to(device)
    
    # wandb에 모델 구조 기록
    wandb.watch(generator, log="all", log_freq=log_freq)
    wandb.watch(discriminator, log="all", log_freq=log_freq)
    
    # 옵티마이저 설정
    g_optimizer = optim.Adam(generator.parameters(), lr=0.00025, betas=(0, 0.99))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.00015, betas=(0, 0.99))
    
    # 결과 저장 디렉토리 생성
    os.makedirs(checkpoint_dir, exist_ok=True)
    samples_dir = os.path.join(checkpoint_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    # 메트릭 로거 초기화
    metrics_logger = MetricsLogger(checkpoint_dir)
    
    # 시작 에포크 초기화
    start_epoch = 0
    
    # 체크포인트에서 복구
    if resume is not None:
        print(f"Loading checkpoint from {resume}")
        checkpoint = torch.load(resume)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
        
        # wandb에 체크포인트 복구 기록
        wandb.log({"resume_from_epoch": start_epoch})
    
    # 데이터로더 설정
    train_dataloader = get_dataloader(batch_size=batch_size, num_workers=num_workers)
    eval_dataloader = get_dataloader(batch_size=batch_size, num_workers=num_workers)

    pl_mean = 0
    # 훈련 루프
    for epoch in range(start_epoch, n_epochs):
        torch.cuda.empty_cache()
        generator.train()
        discriminator.train()
        
        epoch_g_losses = []
        epoch_d_losses = []
        
        # epoch 진행률을 wandb에 기록
        wandb.log({"epoch": epoch, "progress": epoch/n_epochs})
        
        # Training loop 수정
        for i, (real_images, real_labels) in enumerate(train_dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            real_labels = real_labels.to(device)
            real_images.requires_grad = True  # Enable gradients for R1
            
            # real_labels를 슈퍼클래스로 매핑
            real_labels = torch.tensor(
                [intra_fids_list[label.item()] for label in real_labels],
                device=device
            ).long()

            z = torch.randn(batch_size, latent_dim, device=device)

            # Generate fake images
            w_styles = style_vectorizer(z)
            spatial_noise = image_noise(batch_size, 32, device)  # Spatial noise
            fake_images = generator(w_styles, real_labels, spatial_noise)
           
            # DiffAug 적용
            real_images_aug = apply_diffaug(real_images, policy=diffaug_policy)
            fake_images_aug = apply_diffaug(fake_images, policy=diffaug_policy)
           
            # Discriminator 학습
            d_optimizer.zero_grad()

            d_real, dr_q_loss = discriminator(real_images_aug, real_labels)
            d_fake, df_q_loss = discriminator(fake_images_aug.detach(), real_labels)
           
            # WGAN-GP Loss
            gp = gradient_penalty(discriminator, real_images_aug, fake_images_aug.detach(), device, real_labels)

            hinge_loss = (F.relu(1 + d_real) + F.relu(1 - d_fake)).mean()
            quantize_loss = (df_q_loss + dr_q_loss).mean()

            d_loss = hinge_loss + quantize_loss + 10.0 * gp
            d_loss.backward()
            
            # Gradient Clipping 적용
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1)
            d_optimizer.step()
            epoch_d_losses.append(d_loss.item())  # 이 부분 추가
            
            if i % 5 == 0:
                g_optimizer.zero_grad()
                d_fake, fake_class_pred = discriminator(fake_images_aug, real_labels)
                g_loss = d_fake.mean()
                if (epoch >= 100) and (i % 30 == 0):
                    pl_lengths = calc_pl_lengths(w_styles, fake_images)
                    avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())

                    if pl_mean:
                        pl_loss = ((pl_lengths - pl_mean) ** 2).mean()
                        if not torch.isnan(pl_loss):
                            g_loss = g_loss + pl_loss
                    pl_mean = pl_mean * 0.99 + avg_pl_length * 0.01
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 1)
                g_optimizer.step()
                epoch_g_losses.append(g_loss.item()) 
            
            # 로깅
            if i % log_freq == 0:
                step_metrics = {
                    'train/d_loss': d_loss.item(),
                    'train/g_loss': g_loss.item(),
                    'train/gp': gp.item(),
                    'train/d_real': torch.mean(d_real).item(),
                    'train/d_fake': torch.mean(d_fake).item(),
                    'epoch': epoch,
                    'step': i
                }
                wandb.log(step_metrics)
                
                print(f'Epoch [{epoch}/{n_epochs}] Step [{i}/{len(train_dataloader)}] '
                      f'd_loss: {d_loss.item():.4f} g_loss: {g_loss.item():.4f}')
                
        save_generated_images(generator, style_vectorizer, epoch+1, samples_dir, SUPERCLASS_NAMES, intra_fids_list, device=device)
        # eval_freq 에포크마다 평가 수행
        if (epoch + 1) % eval_freq == 0:
            print(f"\nEvaluating epoch {epoch+1}...")
            eval_metrics = evaluate_epoch(
                generator=generator,
                style_vectorizer=style_vectorizer,
                real_dataloader=eval_dataloader,
                n_eval_samples=n_eval_samples,
                batch_size=eval_batch_size,
                device=device
            )
                    
            # 손실값 추가
            eval_metrics['d_loss'] = np.mean(epoch_d_losses)
            eval_metrics['g_loss'] = np.mean(epoch_g_losses)
        
            # 메트릭 로깅
            metrics_logger.update(epoch, eval_metrics)
            
            print(f"Epoch {epoch+1} Metrics:")
            print(f"Inception Score: {eval_metrics['inception_score']['mean']:.2f} ± {eval_metrics['inception_score']['std']:.2f}")
            print(f"FID Score: {eval_metrics['fid']:.2f}")
            print(f"Intra-FID Score: {eval_metrics['intra_fid']['mean']:.2f}")
            print(f"Average D Loss: {np.mean(epoch_d_losses):.4f}")
            print(f"Average G Loss: {np.mean(epoch_g_losses):.4f}\n")
        else:
            # 평가를 건너뛸 때는 손실값만 기록
            metrics_logger.update(epoch, {
                'd_loss': np.mean(epoch_d_losses),
                'g_loss': np.mean(epoch_g_losses)
            })
        
        if (best_metrics[3] > eval_metrics['intra_fid']['mean']):
            best_metrics = [eval_metrics['inception_score']['mean'], eval_metrics['inception_score']['std'], eval_metrics['fid'], eval_metrics['intra_fid']['mean']]

        # 체크포인트 저장
        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            checkpoint_data = {
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'epoch': epoch,
                'metrics': eval_metrics if (epoch + 1) % eval_freq == 0 else None
            }
            # torch.save(checkpoint_data, checkpoint_file) # If you want to save model, use this
            print(f"Saved checkpoint to {checkpoint_file}")
    
    print("Best_Metrics")
    print(f'IS: {best_metrics[0]}±{best_metrics[1]}')
    print(f'FID: {best_metrics[2]}')
    print(f'Intra-FID: {best_metrics[3]}')
    return generator, discriminator