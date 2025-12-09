import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models import UNet
from data.dataset import ExposureDataset
from utils import MetricsCalculator

def evaluate(model, loader, metrics_calc, device, save_dir=None):
    model.eval()
    all_metrics = []
    
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for idx, (inputs, gt) in enumerate(tqdm(loader, desc="Evaluating")):
            inputs, gt = inputs.to(device), gt.to(device)
            pred = model(inputs)
            
            for i in range(pred.size(0)):
                metrics = metrics_calc.compute_all(pred[i:i+1], gt[i:i+1])
                all_metrics.append(metrics)
                
                if save_dir:
                    img = pred[i].permute(1,2,0).cpu().numpy()
                    img = (img * 255).astype(np.uint8)
                    cv2.imwrite(
                        str(save_dir / f"{idx*loader.batch_size + i:04d}.png"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    )
    
    return metrics_calc.aggregate_metrics(all_metrics)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    
    test_set = ExposureDataset(args.test_input, args.test_gt)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    metrics_calc = MetricsCalculator(device)
    save_dir = Path(args.results_dir) if args.save_predictions else None
    
    results = evaluate(model, test_loader, metrics_calc, device, save_dir)
    
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"PSNR  : {results['psnr_mean']:.4f} ± {results['psnr_std']:.4f}")
    print(f"SSIM  : {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
    print(f"LPIPS : {results['lpips_mean']:.4f} ± {results['lpips_std']:.4f}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--test_input', required=True)
    parser.add_argument('--test_gt', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--save_predictions', action='store_true')
    main(parser.parse_args())
