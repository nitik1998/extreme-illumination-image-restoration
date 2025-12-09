import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import argparse
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models import UNet
from data.dataset import get_data_loaders
from utils import MetricsCalculator, setup_logger, log_metrics, compute_psnr

def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, running_psnr = 0.0, 0.0
    
    pbar = tqdm(loader, desc="Training")
    for inputs, gt in pbar:
        inputs, gt = inputs.to(device, non_blocking=True), gt.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, gt)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        running_psnr += compute_psnr(outputs, gt)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return running_loss / len(loader), running_psnr / len(loader)

def main(args):
    logger = setup_logger('unet_training', 'logs')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    train_loader, test_loader = get_data_loaders(
        args.train_input, args.train_gt,
        args.test_input, args.test_gt,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    model = UNet().to(device)
    logger.info(f"Parameters: {model.count_parameters():,}")
    
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = GradScaler()
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss, train_psnr = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        
        scheduler.step(train_loss)
        epoch_time = time.time() - start_time
        
        metrics = {'loss': train_loss, 'psnr': train_psnr, 'time': epoch_time}
        log_metrics(logger, epoch, metrics)
        
        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'checkpoints/best_unet.pth')
            logger.info("✓ Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping")
                break
    
    logger.info("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_input', required=True)
    parser.add_argument('--train_gt', required=True)
    parser.add_argument('--test_input', required=True)
    parser.add_argument('--test_gt', required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--patience', type=int, default=5)
    main(parser.parse_args())
