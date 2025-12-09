import torch
import torch.nn.functional as F
from math import log10
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_psnr(pred, gt):
    mse = F.mse_loss(pred, gt).item()
    return 100.0 if mse == 0 else 10 * log10(1 / mse)

class MetricsCalculator:
    def __init__(self, device='cuda'):
        self.device = device
        self.lpips_fn = None
    
    def _load_lpips(self):
        if self.lpips_fn is None:
            try:
                import lpips
                self.lpips_fn = lpips.LPIPS(net='vgg').to(self.device)
                print("✓ LPIPS loaded")
            except ImportError:
                print("⚠️  Install lpips: pip install lpips")
    
    def compute_psnr(self, pred, gt):
        pred = pred.squeeze().permute(1, 2, 0).cpu().numpy()
        gt = gt.squeeze().permute(1, 2, 0).cpu().numpy()
        return peak_signal_noise_ratio(gt, pred, data_range=1.0)
    
    def compute_ssim(self, pred, gt):
        pred = pred.squeeze().permute(1, 2, 0).cpu().numpy()
        gt = gt.squeeze().permute(1, 2, 0).cpu().numpy()
        return structural_similarity(gt, pred, channel_axis=2, data_range=1.0)
    
    def compute_lpips(self, pred, gt):
        self._load_lpips()
        return 0.0 if self.lpips_fn is None else self.lpips_fn(pred, gt).item()
    
    def compute_all(self, pred, gt):
        return {
            'psnr': self.compute_psnr(pred, gt),
            'ssim': self.compute_ssim(pred, gt),
            'lpips': self.compute_lpips(pred, gt)
        }
    
    def aggregate_metrics(self, metrics_list):
        psnr_vals = [m['psnr'] for m in metrics_list]
        ssim_vals = [m['ssim'] for m in metrics_list]
        lpips_vals = [m['lpips'] for m in metrics_list]
        return {
            'psnr_mean': np.mean(psnr_vals),
            'psnr_std': np.std(psnr_vals),
            'ssim_mean': np.mean(ssim_vals),
            'ssim_std': np.std(ssim_vals),
            'lpips_mean': np.mean(lpips_vals),
            'lpips_std': np.std(lpips_vals),
        }
