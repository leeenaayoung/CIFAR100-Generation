import argparse
import torch
import os
from datetime import datetime
import wandb
from stylegan2_model import StyleGAN2Generator, StyleGAN2Discriminator
from train import train_stylegan2

def parse_args():
    parser = argparse.ArgumentParser(description='Train StyleGAN2 on CIFAR-100')
    
    # wandb 관련 설정
    parser.add_argument('--wandb-project', type=str, default='stylegan2_superclass',
                      help='WandB project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                      help='WandB entity name')
    parser.add_argument('--wandb-name', type=str, default=None,
                      help='WandB run name')
    
    # 기존 파라미터들...
    parser.add_argument('--n-epochs', type=int, default=1000,
                      help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=64,
                      help='Training batch size')
    parser.add_argument('--latent-dim', type=int, default=512,
                      help='Dimension of latent space')
    parser.add_argument('--lr', type=float, default=0.0002,
                      help='Learning rate')
    
    # 평가 관련 파라미터
    parser.add_argument('--n-eval-samples', type=int, default=50000,
                      help='Number of samples to generate for evaluation')
    parser.add_argument('--eval-batch-size', type=int, default=400,
                      help='Batch size for evaluation')
    parser.add_argument('--eval-freq', type=int, default=100,
                      help='Frequency of evaluation (epochs)')
    
    # 하드웨어 설정
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda or cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Number of data loading workers')
    
    # 체크포인트 관련 설정
    parser.add_argument('--checkpoint-dir', type=str, default='savepoints',
                      help='Directory to save checkpoints')
    parser.add_argument('--checkpoint-freq', type=int, default=20,
                      help='Frequency of saving checkpoints (epochs)')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume training from')
    
    # 로깅 관련 설정
    parser.add_argument('--log-freq', type=int, default=100,
                      help='Frequency of logging training progress (steps)')
    
    return parser.parse_args()

def main():
    torch.cuda.empty_cache()
    args = parse_args()
    
    # wandb 초기화
    run_name = args.wandb_name if args.wandb_name else f"stylegan2-cifar100-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args)
    )
    
    # Device 설정
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 체크포인트 디렉토리 생성
    experiment_name = wandb.run.name
    checkpoint_path = os.path.join(args.checkpoint_dir, experiment_name)
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # 훈련 설정 저장
    training_config = vars(args)
    with open(os.path.join(checkpoint_path, 'config.txt'), 'w') as f:
        for key, value in training_config.items():
            f.write(f'{key}: {value}\n')
    
    # 모델 훈련
    print("Starting training...")
    generator, discriminator = train_stylegan2(
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        device=device,
        checkpoint_dir=checkpoint_path,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        resume=args.resume,
        log_freq=args.log_freq,
        lr=args.lr,
        num_workers=args.num_workers,
        n_eval_samples=args.n_eval_samples,
        eval_batch_size=args.eval_batch_size
    )
    
    print("\nTraining completed!")
    
    # 최종 모델 저장
    final_checkpoint = os.path.join(checkpoint_path, 'final_model.pt')
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
    }, final_checkpoint)
    print(f"Final model saved to {final_checkpoint}")
    
    # wandb 종료
    wandb.finish()

if __name__ == '__main__':
    main()