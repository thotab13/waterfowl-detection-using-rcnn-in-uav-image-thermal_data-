# Computer Vision – Object Detection Assignment
# Waterfowl Detection in UAV Thermal Imagery using Faster R-CNN

**Submitted by:**
**Submitted by:**
* Thota Bhuvana Chandra (10001026)
* Rachitha Chandra Bhanu (10000726)
* Sai Surya Mada (10000982)

---

## 📌 Project Overview
This project implements an automated **waterfowl detection system** using **Faster R-CNN (ResNet-50 FPN)** trained on thermal UAV imagery. The system detects individual birds for wildlife monitoring and ecological surveys.

### 🔹 Key Features
* Thermal-only detection system 
* Faster R-CNN with ResNet-50 FPN backbone
* Custom PyTorch dataset for positive + negative images
* Thermal-specific data augmentation (horizontal flip, Gaussian blur)
* Evaluation using Precision, Recall, F1-score, and mAP@0.5
* Visualization of TP, FP, FN detections

---

## 📌 Prerequisites
Install necessary libraries:

```bash
pip install torch torchvision matplotlib pandas pillow numpy
```

---

## 📁 Repository Structure
```
├── waterfowl_model.ipynb                       # Notebook containing full pipeline
├── README.md                        # This file
│
├── dataset/                         # UAV Thermal Dataset
│   ├── 01_Positive_Image/          # Thermal images with birds
│   ├── 02_Groundtruth_Label/       # Bounding box annotations (CSV)
│   └── 03_Negative_Images/         # Background images without birds
│
├── outputs/                         # Saved model & results
│   ├── faster_rcnn_waterfowl.pth   # Trained model weights
```

---

## 📌 Training Pipeline
1. Load thermal images and CSV annotations.
2. Convert grayscale → 3-channel tensor for Faster R-CNN.
3. Apply augmentations:
   * Random Horizontal Flip
   * Gaussian Blur
4. Split dataset into **60% train**, **20% val**, **20% test**.
5. Train Faster R-CNN with:
   * Optimizer: AdamW (lr=1e-4)
   * Scheduler: StepLR (gamma=0.1 every 5 epochs)
   * Epochs: 15

---

## 📌 Model Architecture
### Faster R-CNN with ResNet-50 FPN
* **Backbone:** ResNet-50 with Feature Pyramid Network
* **RPN:** Generates region proposals
* **RoIAlign:** Extracts features for each proposal
* **Two Heads:**
  - Classification Head (background vs waterfowl)
  - Bounding Box Regression Head

---

## 📌 Evaluation Metrics
The final trained model achieved:

* **Precision:** 0.888
* **Recall:** 0.920
* **F1-score:** 0.904
* **mAP@0.5:** ~0.901

These results indicate strong detection performance on thermal UAV imagery.

---

## 📌 Visualizations
The project provides visual examples of:

* **True Positives (TP)** – correctly detected birds
* **False Positives (FP)** – incorrect detections on background
* **False Negatives (FN)** – missed birds

Bounding boxes:
* Green = Ground Truth
* Blue = True Positive
* Red = False Positive

---

## 📌 Saving & Loading the Model
```python
# Save model
torch.save(model.state_dict(), 'faster_rcnn_waterfowl.pth')

# Load model
model = get_model(2)
model.load_state_dict(torch.load('faster_rcnn_waterfowl.pth'))
model.to(DEVICE)
```

---

## 📌 Future Improvements
* Add anchor size tuning for tiny birds
* Introduce thermal image normalization
* Try fusion of thermal + RGB modalities

---

