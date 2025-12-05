# Module 3 Report  
### CNN Model Development

## Introduction  
This module built a simple CNN to classify product images into stock code categories.

## High-Level Flow  
1. Load and label scraped images  
2. Train a CNN from scratch  
3. Save model weights  
4. Predict class for unseen images

## Description of Work  
- Designed a three-layer CNN with ReLU, MaxPool, and Linear layers  
- Used CrossEntropy loss and Adam optimiser  
- Trained for three epochs to keep runtime short  
- Model saved as cnn_model.pth  
- Detection endpoint uses the model output to trigger vector search

## Key Decisions  
- Used a compact CNN to avoid long training  
- Normalised images to 224x224 RGB  
- Label mapping stored inside the model file

## Challenges and Solutions  
- Limited training samples solved by keeping classes balanced  
- Some classes looked visually similar; augmentation improved robustness  
- Early failures came from incorrect folder structure, fixed by grouping images by stock code

## Conclusion  
Module 3 delivered an image detection pipeline that works end-to-end with the recommendation service.

## References  
- PyTorch  
- Torchvision transforms  
- Flask integration
