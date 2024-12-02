import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        out = F.linear(input, self.weight * self.scale, 
                      bias=self.bias * self.lr_mul if self.bias is not None else None)
        return out

class StyleGAN2Generator(nn.Module):
    def __init__(self, latent_dim=512, n_mlp=8, n_classes=20):
        super().__init__()
        self.style_dim = latent_dim
        self.n_classes = n_classes
        
        # Embedding layer for class conditioning
        self.class_embedding = nn.Embedding(n_classes, latent_dim)
        
        # Style mapping network
        layers = []
        for i in range(n_mlp):
            layers.append(EqualLinear(latent_dim, latent_dim, lr_mul=0.01))
            layers.append(nn.LeakyReLU(0.2))
        
        self.style = nn.Sequential(*layers)
        
        # Synthesis network
        self.conv_layers = nn.ModuleList([
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.Conv2d(512, 256, 3, 1, 1, bias=False),
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.Conv2d(64, 3, 3, 1, 1, bias=False)
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm2d(512),
            nn.BatchNorm2d(256),
            nn.BatchNorm2d(128),
            nn.BatchNorm2d(64)
        ])

    def forward(self, z, class_labels):
        # Get class embeddings
        c = self.class_embedding(class_labels)
        
        # Combine noise and class embeddings
        w = z + c
        
        # Style mapping network
        w = self.style(w)
        
        # Synthesis network
        x = w.view(-1, self.style_dim, 1, 1)
        
        for i, (conv, bn) in enumerate(zip(self.conv_layers[:-1], self.batch_norms)):
            x = conv(x)
            x = bn(x)
            x = F.leaky_relu(x, 0.2)
            if i < 3:  # Upsampling for first 3 layers
                x = F.interpolate(x, scale_factor=2, mode='nearest')
        
        x = self.conv_layers[-1](x)
        x = torch.tanh(x)
        
        return x

class StyleGAN2Discriminator(nn.Module):
   def __init__(self, n_classes=20):
       super().__init__()
       self.class_embedding = nn.Embedding(n_classes, 512)
       
       self.conv_layers = nn.ModuleList([
           nn.Conv2d(3, 64, 3, 2, 1, bias=False),
           nn.Conv2d(64, 128, 3, 2, 1, bias=False),
           nn.Conv2d(128, 256, 3, 2, 1, bias=False),
           nn.Conv2d(256, 512, 3, 2, 1, bias=False)
       ])
       
       self.batch_norms = nn.ModuleList([
           nn.BatchNorm2d(64),
           nn.BatchNorm2d(128),
           nn.BatchNorm2d(256),
           nn.BatchNorm2d(512)
       ])
       
       self.final_layer = nn.Linear(512 * 2 * 2 + 512, 1)
       # 클래스 분류를 위한 레이어 추가
       self.classifier = nn.Linear(512 * 2 * 2, n_classes)

   def forward(self, x, labels=None):
       for conv, bn in zip(self.conv_layers, self.batch_norms):
           x = conv(x)
           x = bn(x)
           x = F.leaky_relu(x, 0.2)
       
       features = x.view(-1, 512 * 2 * 2)
       
       # 클래스 예측
       class_pred = self.classifier(features)
       
       # 판별값 계산
       if labels is not None:
           c = self.class_embedding(labels)
           disc_in = torch.cat([features, c], dim=1)
       else:
           disc_in = features
           
       disc_out = self.final_layer(disc_in)
       
       return disc_out, class_pred