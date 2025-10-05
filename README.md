# computer-vision

A collection of computer vision lab assignments and experiments covering fundamental image processing techniques, spatial and frequency domain operations, and imaging geometry concepts.


## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Conda or Miniconda installed

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

### Lab 2: Advanced Computer Vision Techniques
*[Lab 2 content will be added as it becomes available]*

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
│   └── Lab1 Manual.pdf    # Instructions
├── lab2/                   # Laboratory 2 files (TBD)
├── environment.yml         # Conda environment specification
├── LICENSE                 # MIT License
└── README.md              # This file
```
