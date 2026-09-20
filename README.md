# Face Recognition Attendance System

A face-recognition-based attendance system built with classical Machine Learning. The project identifies people from face images using **PCA (Eigenfaces)** for feature extraction and an **SVM** classifier, then automatically logs each recognized person into an attendance sheet (`attendance.csv`).

## Project Overview

The goal of this project is to simulate an automatic attendance system: given a face image, the model identifies the person and marks them present once, with the date and time.

The project workflow includes:

- Loading and exploring the face images dataset
- Flattening images into feature vectors
- Splitting the data into training and testing sets (stratified)
- Dimensionality reduction with PCA (Eigenfaces)
- Training an SVM classifier
- Evaluating model performance
- Generating an attendance sheet from the recognized faces

## Dataset

The project uses the **AT&T Database of Faces** (also known as the ORL dataset), available on Kaggle:
[att-database-of-faces](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces)

| Property | Value |
|----------|------:|
| People (classes) | 40 |
| Images per person | 10 |
| Total images | 400 |
| Image size | 92 × 112 pixels (grayscale) |
| Features per image (flattened) | 10,304 |

Each person has their own folder (`s1` … `s40`) containing 10 `.pgm` images.

## Technologies & Libraries

- Python
- NumPy
- Pandas
- Pillow (PIL)
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Methodology

### 1. Data Loading

Every image is opened in grayscale and flattened into a vector of 10,304 pixel values. The folder name (`s1`, `s2`, …) is used as the person's ID (label).

### 2. Train / Test Split

The data is split **80% / 20%** using a stratified split (`random_state=42`), so every person appears in both sets:

- Training images: **320**
- Test images: **80**

### 3. Feature Extraction — PCA (Eigenfaces)

PCA with `n_components=100` and `whiten=True` reduces each image from 10,304 features to 100 principal components. PCA is fitted on the training set only, then applied to the test set.

### 4. Classification — SVM

A Support Vector Machine with an RBF kernel is trained on the PCA features:

```python
SVC(kernel='rbf', class_weight='balanced', C=1000, gamma=0.001, probability=True)
```

## Results

| Metric | Value |
|--------|------:|
| Test accuracy | **98.75%** |
| Test samples | 80 |
| Misclassified | 1 |

The model correctly identified 79 of the 80 test images; the single error was a face of `s10` predicted as `s29`.

## Attendance System

After training, the recognized faces are used to build an attendance sheet:

1. Each test face is passed to the classifier to predict the person's ID.
2. If the person has not been marked yet, a row is written to `attendance.csv` with the **Student ID**, **Date**, and **Time**.
3. If the person was already marked, the system prints a warning instead of adding a duplicate.

In the notebook run, all **40 people** were marked present exactly once.

Sample of `attendance.csv`:

| Student_ID | Date | Time |
|------------|------|------|
| s29 | 2026-09-16 | 09:33:16 |
| s5 | 2026-09-16 | 09:33:16 |
| s26 | 2026-09-16 | 09:33:16 |

## Project Workflow

```text
Face Images (92×112, grayscale)
   ↓
Flatten to 10,304 features
   ↓
Train / Test Split (80/20, stratified)
   ↓
PCA (100 components, whitened)
   ↓
SVM Classifier (RBF kernel)
   ↓
Model Evaluation
   ↓
Face Recognition → Attendance Sheet (CSV)
```

## Project Structure

```text
Face-Recognition-Attendance-System/
│
├── Data/
│   └── att-database-of-faces/
│       ├── s1/
│       ├── s2/
│       └── ... (s40)
├── face-recognition-attendance.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/AbdelazizElshourbgy-ui/Face-Recognition-Attendance-System.git
```

### 2. Navigate to the project directory

```bash
cd Face-Recognition-Attendance-System
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Download the [AT&T Database of Faces](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces) from Kaggle and extract it so the folders `s1` … `s40` are inside:

```text
Data/att-database-of-faces/
```

### 5. Open the notebook

```bash
jupyter notebook
```

Then open:

```text
face-recognition-attendance.ipynb
```

Run the notebook from the project's root folder so the relative `Data/` path works.

## Limitations

- The system uses **classical ML (PCA + SVM)**, not deep learning.
- It performs **closed-set identification**: it can only recognize the 40 people it was trained on, and always predicts one of them.
- The attendance sheet is generated from the dataset's test images as a simulation, not from a live camera feed.
- The dataset was captured under controlled conditions, so performance on real-world images with different lighting and angles will likely be lower.

## Skills Demonstrated

- Computer Vision Basics
- Face Recognition
- Image Processing
- Dimensionality Reduction (PCA / Eigenfaces)
- Support Vector Machines (SVM)
- Machine Learning Classification
- Model Evaluation
- Python
- NumPy
- Pandas
- Scikit-learn

## Author

**Abdelaziz Elshourbgy**

Computer Science Student | Data Analysis | Machine Learning | AI

- GitHub: [AbdelazizElshourbgy-ui](https://github.com/AbdelazizElshourbgy-ui)
- Kaggle: [abdelazizelshourbgy](https://www.kaggle.com/abdelazizelshourbgy)
