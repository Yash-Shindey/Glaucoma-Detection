# Glaucoma Detection via Deep Learning CNN Models

## Project Overview
This project implements multiple convolutional neural network (CNN) architectures to detect glaucoma from retinal fundus images. The system analyzes optical cup-to-disc ratio (CDR) and other features to classify images as either glaucomatous or healthy with high accuracy. The implementation includes preprocessing pipelines, model training, and comprehensive evaluation metrics.

## Dataset
- **Source**: ORIGA (Online Retinal fundus Image database for Glaucoma Analysis)
- **Size**: 650 retinal fundus images with ophthalmologist annotations
- **Labels**: Binary classification (glaucoma/non-glaucoma)
- **Metadata**: ExpCDR (Expected Cup to Disc Ratio) values for each image
- **Distribution**: 79 glaucomatous images (12.15%) and 571 normal images (87.85%)

## Technical Implementation

### Preprocessing Pipeline
- **Image Enhancement**:
  - Gamma correction (γ=0.4) for contrast improvement
  - CLAHE (Contrast Limited Adaptive Histogram Equalization) with clip limit 2.0
  - Black padding removal with threshold-based detection
  - Gaussian blur (5×5 kernel) for noise reduction
  
- **Segmentation Techniques**:
  - K-means clustering (6 clusters) for disc and cup segmentation
  - Morphological operations for noise removal and boundary enhancement
  - Cup-to-disc ratio calculation using bounding box detection

- **Data Preparation**:
  - Image resizing to 448×448 pixels (224×224 for some models)
  - 80/20 train-test split with stratification
  - Data augmentation (horizontal/vertical flips, normalization)

### Model Architectures Implemented
- **VGG16**: Achieved 85% accuracy with BCE loss and Adam optimizer (lr=0.001)
- **DenseNet121**: Implemented with transfer learning and fine-tuned classifier
- **InceptionV3**: Used with auxiliary classifier for improved gradient flow
- **EfficientNet-B0**: Optimized for parameter efficiency with 77% precision

### Training Strategy
- **Batch Size**: 16 for all models
- **Epochs**: 10 for consistent comparison
- **Optimizer**: AdamW with learning rate 5e-5
- **Loss Function**: Binary Cross-Entropy with Logits
- **Regularization**: Dropout (0.5) to prevent overfitting

## Results & Performance

### Metrics Comparison
| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| VGG16 | 85% | 0.81 | 0.64 | 0.72 |
| DenseNet121 | 68% | 0.70 | 0.70 | 0.70 |
| InceptionV3 | 75% | 0.63 | 0.54 | 0.58 |
| EfficientNet | 72% | 0.77 | 0.74 | 0.75 |

### Key Findings
- The VGG16 model achieved the highest overall accuracy (85%)
- EfficientNet provided the best balance between precision and recall (F1 score: 0.75)
- Model performance correlated with cup-to-disc ratio measurements, confirming clinical relevance
- K-means segmentation effectively isolated the optic disc and cup regions for analysis

## Future Improvements
- Implement ensemble methods combining the strengths of multiple architectures
- Explore additional data augmentation techniques for improved generalization
- Incorporate explainable AI techniques for highlighting clinically relevant features
- Extend the analysis to multi-class classification for glaucoma staging

## Technologies Used
- **Python**: Core programming language
- **PyTorch**: Deep learning framework
- **OpenCV**: Image processing and segmentation
- **scikit-learn**: Evaluation metrics and K-means clustering
- **albumentations**: Advanced image augmentation
- **matplotlib/seaborn**: Visualization of results

This project demonstrates the application of modern deep learning techniques to medical image analysis, potentially aiding in early glaucoma detection and reducing the burden of manual screening by ophthalmologists.
