# Multi-Exposure & Low-Light Image Enhancement

A comprehensive benchmark comparing different deep learning models for multi-exposure and low-light image enhancement.

## 🌟 Features

- **Multiple Models**: U-Net and Zero-DCE++ implementations
- **Complete Pipeline**: Training, evaluation, and inference scripts
- **Optimized**: Efficient data loading, mixed precision training, and GPU optimization
- **Modular**: Clean, maintainable code structure
- **Benchmarking**: PSNR, SSIM, and LPIPS metrics

## 📊 Results

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|-------|--------|--------|---------|
| Zero-DCE++ | 11.49 | 0.668 | 0.219 |
| **U-Net** | **18.44** | **0.782** | **0.209** |

U-Net achieves **+6.95 dB PSNR** improvement over Zero-DCE++!

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/multi-exposure-enhancement.git
cd multi-exposure-enhancement
pip install -r requirements.txt
```

### Dataset Structure

```
Dataset/
├── train/
│   ├── INPUT_IMAGES/
│   └── GT_IMAGES/
└── test/
    ├── INPUT_IMAGES/
    └── GT_IMAGES/
```

### Training

**Train U-Net:**
```bash
python scripts/train_unet.py \
    --train_input /path/to/train/INPUT_IMAGES \
    --train_gt /path/to/train/GT_IMAGES \
    --test_input /path/to/test/INPUT_IMAGES \
    --test_gt /path/to/test/GT_IMAGES \
    --epochs 30 \
    --batch_size 32
```

**Evaluate:**
```bash
python scripts/evaluate.py \
    --model unet \
    --checkpoint checkpoints/best_unet.pth \
    --test_input /path/to/test/INPUT_IMAGES \
    --test_gt /path/to/test/GT_IMAGES \
    --save_predictions
```

**Inference:**
```bash
python scripts/inference.py \
    --model unet \
    --checkpoint checkpoints/best_unet.pth \
    --input_dir /path/to/images \
    --output_dir results/predictions
```

## 📁 Project Structure

```
multi-exposure-enhancement/
├── models/              # Model architectures
│   ├── unet.py         # U-Net implementation
│   └── zero_dce.py     # Zero-DCE++ wrapper
├── data/               # Data loading
│   ├── dataset.py      # Dataset class
│   └── transforms.py   # Augmentations
├── utils/              # Utilities
│   ├── metrics.py      # PSNR, SSIM, LPIPS
│   ├── logger.py       # Logging utilities
│   └── visualize.py    # Visualization
├── scripts/            # Training & evaluation
│   ├── train_unet.py   # Train U-Net
│   ├── evaluate.py     # Evaluate models
│   └── inference.py    # Run inference
├── configs/            # Configuration files
├── notebooks/          # Jupyter notebooks
├── tests/              # Unit tests
└── docs/               # Documentation
```

## 🔧 Configuration

Edit `configs/train_config.py` or use command-line arguments:

```python
# Key parameters
batch_size = 32          # Adjust based on GPU memory
learning_rate = 1e-4
epochs = 30
patience = 5             # Early stopping
```

## 📈 Training Details

### U-Net
- **Architecture**: 3-level encoder-decoder with skip connections
- **Parameters**: ~7.8M
- **Batch Size**: 32 (optimized for A100 GPU)
- **Training Time**: ~2 hours on 17,675 images
- **Throughput**: ~165 images/sec

### Hardware Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (16GB+ recommended)
- **RAM**: 16GB+ system memory
- **Storage**: 10GB+ for dataset

## 🎯 Metrics

- **PSNR**: Peak Signal-to-Noise Ratio (higher is better)
- **SSIM**: Structural Similarity Index (higher is better)  
- **LPIPS**: Learned Perceptual Image Patch Similarity (lower is better)

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{multi-exposure-enhancement,
  author = {Your Name},
  title = {Multi-Exposure Image Enhancement Benchmark},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/YOUR_USERNAME/multi-exposure-enhancement}
}
```

## 🙏 Acknowledgments

- [Zero-DCE++](https://github.com/Li-Chongyi/Zero-DCE_extension) by Li et al.
- U-Net architecture from Ronneberger et al.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## 📧 Contact

For questions or issues, please open a GitHub issue or contact [your.email@example.com](mailto:your.email@example.com)
