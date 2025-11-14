# computer-vision

A collection of computer vision lab assignments and experiments covering fundamental image processing techniques, spatial and frequency domain operations, image binarization methods, stereo vision, and imaging geometry concepts. Includes both Python Jupyter notebooks and MATLAB implementations.


## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Conda or Miniconda installed
- MATLAB (for Lab 2 assignments)

### Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/computer-vision.git
   cd computer-vision
   ```

2. **Create and activate the conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate computer-vision
   ```

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

### Dependencies

The project uses the following key libraries:
- **OpenCV** - Computer vision and image processing
- **NumPy** - Numerical computing
- **Matplotlib** - Plotting and visualization
- **SciPy** - Scientific computing
- **scikit-image** - Image processing algorithms
- **Pillow** - Image I/O operations
- **Pandas** - Data manipulation
- **Seaborn** - Statistical visualization

## 📚 Laboratory Assignments

### Lab 1: Fundamental Image Processing
**Topics Covered:**
- **Point Processing Operations**
  - Contrast stretching
  - Histogram equalization
- **Spatial Filtering**
  - Gaussian filtering for noise removal
  - Median filtering techniques
- **Frequency Domain Operations**
  - Fourier transforms
  - Frequency domain filtering
- **Imaging Geometry**
  - Geometric transformations
  - Image warping and rectification

**Files:**
- `lab1/lab1.ipynb` - Main notebook with implementations and experiments
- `lab1/corner.m` - MATLAB script for corner detection (reference)
- `lab1/Lab1 Manual.pdf` - Detailed assignment instructions

### Lab 2: Image Binarization and Stereo Vision
**Topics Covered:**
- **Image Thresholding Techniques**
  - Otsu's global thresholding method
  - Niblack's adaptive/local thresholding
  - Grid search parameter optimization
  - Bayesian parameter optimization
- **Performance Evaluation**
  - F-measure calculation
  - Precision and recall metrics
  - Ground truth comparison
- **Stereo Vision**
  - Disparity map computation
  - SSD (Sum of Squared Differences) matching
  - 3D depth estimation from stereo pairs

**Files:**
- `lab2/otsu_thresholding.m` - Otsu's global thresholding implementation
- `lab2/niblack_local_thresholding.m` - Niblack's method with grid search optimization
- `lab2/niblack_bayesian.m` - Niblack's method with Bayesian optimization
- `lab2/stereo_vision.m` - Stereo vision and disparity map computation
- `lab2/assets/` - Document images, ground truths, and stereo image pairs
- `lab2/docs/` - Lab manuals and assignment instructions

## 🛠️ Project Structure

```
computer-vision/
├── assets/                 # Test images and resources
│   ├── book.jpg
│   ├── lena.jpg
│   ├── corridor*.jpg
│   └── ...
├── lab1/                   # Laboratory 1 files
│   ├── lab1.ipynb         # Main notebook
│   ├── corner.m           # MATLAB reference
│   ├── docs/              # Lab documentation
│   └── assets/            # Lab 1 specific images
├── lab2/                   # Laboratory 2 files
│   ├── otsu_thresholding.m           # Otsu's method
│   ├── niblack_local_thresholding.m  # Niblack with grid search
│   ├── niblack_bayesian.m            # Niblack with Bayesian opt
│   ├── stereo_vision.m               # Stereo vision implementation
│   ├── assets/                        # Document images and stereo pairs
│   └── docs/                          # Lab manuals
├── environment.yml         # Conda environment specification
├── LICENSE                 # MIT License
└── README.md              # This file
```
