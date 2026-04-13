# Heartbeat Sound Analysis

A digital signal processing project for cardiac health classification using heartbeat sound analysis. This project implements both traditional machine learning and deep learning approaches to classify heartbeat sounds as healthy or unhealthy.

## Project Overview

This project focuses on mono-dimensional signal processing of audio signals for cardiac health classification. The analysis includes:

- **Signal preprocessing**: Bandpass filtering and peak detection
- **Feature extraction**: MFCC, zero-crossing rate, spectral centroid, and chroma features
- **Classification**: Random Forest and CNN-based approaches
- **Evaluation**: Comprehensive performance metrics and confusion matrices

## Dataset Structure

```
data/
train/
    healthy/     # Healthy heartbeat recordings
    unhealthy/   # Unhealthy heartbeat recordings
val/
    healthy/     # Validation healthy recordings
    unhealthy/   # Validation unhealthy recordings
```

- **Training samples**: 3,240 audio files
- **Validation samples**: 301 audio files
- **Audio format**: WAV files

## Methodology

### 1. Signal Preprocessing
- **Bandpass filtering**: 20-400 Hz frequency range to focus on cardiac frequencies
- **Peak detection**: Identifying individual heartbeat cycles
- **Beat segmentation**: Extracting 0.3-second windows around detected peaks

### 2. Feature Extraction

#### Traditional ML Features:
- **MFCC** (13 coefficients): Mel-frequency cepstral coefficients
- **Zero-crossing rate**: Signal frequency characteristics
- **Spectral centroid**: Spectral brightness measure

#### Deep Learning Features:
- **MFCC images** (40 coefficients): 2D representation for CNN
- **Multi-feature maps**: Combined MFCC, spectral centroid, ZCR, and chroma features
- **Normalization**: Min-max scaling for consistent input ranges

### 3. Classification Models

#### Random Forest Classifier
- **Architecture**: 100 estimators
- **Features**: 15-dimensional feature vector
- **Performance**: Perfect classification (100% accuracy) on validation set

#### Convolutional Neural Network
- **Architecture**: 
  - 3 Conv2D layers with BatchNormalization
  - MaxPooling2D layers
  - Dropout for regularization
  - Dense output layer with sigmoid activation
- **Input shape**: Variable (typically 40×50×1 for MFCC, 54×50×1 for multi-feature)
- **Optimization**: Adam optimizer with learning rate 0.0005
- **Class weighting**: Balanced class weights to handle data imbalance

## Key Results

### Random Forest Performance
- **Accuracy**: 100%
- **Precision**: 100% (both classes)
- **Recall**: 100% (both classes)
- **F1-score**: 100% (both classes)

### CNN Performance
- **Training accuracy**: ~73%
- **Validation accuracy**: ~50-68% (varies with architecture)
- **Challenges**: Overfitting and class imbalance issues

## Technical Implementation

### Dependencies
- `librosa`: Audio processing and feature extraction
- `numpy`: Numerical computations
- `scikit-learn`: Traditional ML algorithms and metrics
- `tensorflow.keras`: Deep learning framework
- `matplotlib`/`seaborn`: Visualization
- `scipy`: Signal processing utilities

### Key Functions
- `bandpass_filter()`: Signal filtering
- `segment_beats()`: Heartbeat cycle detection and segmentation
- `extract_features()`: Traditional feature extraction
- `beat_to_mfcc_image()`: CNN input preparation
- `beat_to_feature_map()`: Multi-feature CNN input preparation

## Usage

1. **Install dependencies**:
   ```bash
   pip install librosa numpy scikit-learn tensorflow matplotlib seaborn scipy
   ```

2. **Run the analysis**:
   - Open `index.ipynb` in Jupyter Notebook
   - Execute cells sequentially to reproduce the analysis

3. **Data preparation**:
   - Ensure audio files are in the correct directory structure
   - Supported format: WAV files

## File Structure

```
Heartbeat Sound Analysis/
README.md                    # This file
index.ipynb                 # Main analysis notebook
data/                       # Dataset directory
  train/                    # Training data
    healthy/                # Healthy heartbeat recordings
    unhealthy/              # Unhealthy heartbeat recordings
  val/                      # Validation data
    healthy/                # Healthy validation recordings
    unhealthy/              # Unhealthy validation recordings
.gitignore                  # Git ignore file
.vscode/                    # VS Code settings
Heartbeat Sound Analysis.code-workspace  # VS Code workspace
```

## Future Improvements

1. **Data augmentation**: Add noise and time-stretching for robustness
2. **Advanced architectures**: Explore ResNet, attention mechanisms
3. **Hyperparameter tuning**: Systematic optimization of model parameters
4. **Cross-validation**: More robust performance evaluation
5. **Real-time processing**: Implementation for live cardiac monitoring
