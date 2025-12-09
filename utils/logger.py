import logging
from pathlib import Path
from datetime import datetime

def setup_logger(name='training', log_dir='logs'):
    Path(log_dir).mkdir(exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fh = logging.FileHandler(f'{log_dir}/{name}_{timestamp}.log')
    fh.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger

def log_metrics(logger, epoch, metrics, prefix=''):
    msg = f"{prefix}Epoch {epoch}: " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
    logger.info(msg)
