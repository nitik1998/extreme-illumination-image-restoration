import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models import UNet

def load_image(path, size=512):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = img.shape[:2]
    img = img.astype(np.float32) / 255.0
    img = cv2.resize(img, (size, size))
    return torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float(), original_size

def save_image(tensor, path, original_size=None):
    img = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    if original_size:
        img = cv2.resize(img, (original_size[1], original_size[0]))
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png'))
    
    print(f"Processing {len(image_files)} images...")
    
    with torch.no_grad():
        for img_path in tqdm(image_files):
            img_tensor, original_size = load_image(img_path)
            img_tensor = img_tensor.to(device)
            
            enhanced = model(img_tensor)
            
            output_path = output_dir / img_path.name
            save_image(enhanced, output_path, original_size if args.preserve_size else None)
    
    print(f"✓ Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--preserve_size', action='store_true')
    main(parser.parse_args())
