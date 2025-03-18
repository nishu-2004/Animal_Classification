# **Animal_Classification**
---

## **Image Augmentation, Dataset Splitting, and Model Training Analysis**  

## **Overview**  
This project consists of two main parts:  

1. **Dataset Preparation** (`splitting_and_blurring.py`)  
   - Applies **blurring and sharpening** augmentations to images.  
   - Splits the dataset into **train (80%) and validation (20%)** sets.  
   - Organizes images in a structured format for training.  

2. **Model Training Analysis**  
   - Evaluates **Accuracy vs. Epoch** to monitor model performance.  
   - Analyzes **Confusion Matrix** to identify misclassified categories.  

---

## **Part 1: Dataset Preparation**  

### **Features**  
 **Automatic dataset discovery**: Finds all `.png`, `.jpg`, `.jpeg` images.  
 **Data Augmentation**:  
   - **5 blurred variations** (random kernel size: 3, 5, 7, 9).  
   - **5 sharpened variations** using a sharpening filter.  
 **Train/Validation Split**: 80-20 random split.  
 **Progress Tracking** with `tqdm`.  

### **Prerequisites**  
Install required packages:  
```bash
pip install opencv-python numpy tqdm
```

### **Usage**  
1️⃣ **Run the Script**  
```bash
python splitting_and_blurring.py
```

2️⃣ **Provide Input and Output Folder Paths**  
- **Input Folder**: Raw dataset location.  
- **Output Folder**: Where processed images will be saved.  

3️⃣ **Output Folder Structure**  
```
output_folder/
│── train/  # Augmented training images (80%)
│── val/    # Augmented validation images (20%)
```

### **Code Breakdown**  
- `get_all_images(input_folder)`: Scans and collects image file paths.  
- `apply_augmentations(image)`: Generates **blurred** and **sharpened** versions.  
- `process_images(input_folder, output_folder)`: Handles **splitting & augmentation**.  

---

## **Part 2: Model Training Analysis**  

### **1. Accuracy vs. Epoch Plot**  
This graph tracks model performance over time.  

- **Train Accuracy (Blue Line)**: Increases steadily, reaching about **90%**.  
- **Validation Accuracy (Orange Line)**: Stabilizes at **81.9%** after fluctuations.  
- **Key Takeaway**:  
   **No major overfitting**, since train-validation accuracy gap is small.  
   **Improvements**: Try **data augmentation, dropout tuning, or more layers**.  

### **2. Confusion Matrix**  
This heatmap highlights **misclassified classes**.  

- **Diagonal Values**: Correctly classified samples.  
- **Off-Diagonal Values**: Misclassified samples (e.g., **dogs misclassified as cats**).  
- **Observations**:  
  - Some animals (like **zebras and cows**) are well-classified due to distinct patterns.  
  - **Potential Fixes**:  
    🔹 Use **class-weighted training**.  
    🔹 Apply **harder augmentations** (contrast changes, rotations).  
    🔹 Add **more data** for frequently misclassified categories.  

---

## **Conclusion**  
- **Dataset Preparation** successfully applies augmentation & dataset splitting.  
- **Model Performance** is **good (81.9% validation accuracy)** but can improve with:  
  1. **CNN architecture tuning** (more filters, different activations).  
  2. **Pre-trained models** (MobileNet, ResNet).  
  3. **Learning Rate Schedulers** to improve convergence.  

 **Next Steps**: Experiment with hyperparameters and augmentation techniques!  

---

## **License**  
MIT License. Free to use and modify.  

---

This README covers both **data processing & model evaluation** in a structured way. Let me know if you need edits! 🔥
