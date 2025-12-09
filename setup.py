from setuptools import setup, find_packages

setup(
    name="extreme-illumination-image-restoration",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "scikit-image>=0.21.0",
        "lpips>=0.1.4",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
    ],
    author="Nitik Jain",
    author_email="nitik1998@gmail.com",
    description="Lightweight U-Net–based RGB exposure correction under extreme illumination conditions",
    keywords="image enhancement, exposure correction, deep learning, computer vision, low light, HDR",
    url="https://github.com/nitik1998/extreme-illumination-image-restoration",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
