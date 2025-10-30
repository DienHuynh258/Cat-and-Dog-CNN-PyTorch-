# 🐱🐶 Cat vs Dog Image Classification with CNN (PyTorch)

A deep learning project that builds and trains a **Convolutional Neural Network (CNN)** from scratch in **PyTorch** to classify images of cats and dogs.
This project is designed for learners who want to understand CNNs conceptually and practically — including model building, training, evaluation, and visualization.

---

## 🌟 Highlights

* Built **custom CNN** using only PyTorch (no pre-trained model)
* **Data augmentation** and normalization for robust learning
* **Early stopping** & **learning rate scheduler** for stable training
* Evaluation using accuracy, loss curves, and visual predictions
* Clean and modular notebook — beginner-friendly but professional-level

---

## 🧠 Model Overview

### Architecture

```python
SimpleCNNPlus(
  (conv1): Conv2d(3, 32, kernel_size=3, padding=1)
  (bn1): BatchNorm2d(32)
  (conv2): Conv2d(32, 64, kernel_size=3, padding=1)
  (bn2): BatchNorm2d(64)
  (conv3): Conv2d(64, 128, kernel_size=3, padding=1)
  (bn3): BatchNorm2d(128)
  (fc1): Linear(128*28*28, 256)
  (dropout): Dropout(p=0.5)
  (fc2): Linear(256, 2)
)
```

**Key Features:**

* **Batch Normalization** after every convolution block
* **Dropout (p=0.5)** before the final classifier
* **ReLU** activation and **MaxPooling** after each conv layer
* Outputs 2 classes → *Cat* or *Dog*

---

## 📂 Dataset

Dataset: [Dogs vs Cats (Kaggle)](https://www.kaggle.com/c/dogs-vs-cats/data)

After extraction, the dataset was reorganized as:

```
data/
├── train/
│   ├── cats/
│   └── dogs/
└── test/
    ├── cats/
    └── dogs/
```

Train data was split:

* **80%** training
* **20%** validation

Image transformations include resizing to `(224, 224)`, normalization, and random flips for augmentation.

---

## ⚙️ Training Configuration

| Parameter     | Value                                      |
| ------------- | ------------------------------------------ |
| Optimizer     | Adam                                       |
| Learning Rate | 0.0005                                     |
| Loss Function | CrossEntropyLoss                           |
| Batch Size    | 32                                         |
| Epochs        | 100 (with early stopping)                  |
| Patience      | 5                                          |
| Scheduler     | ReduceLROnPlateau (factor=0.5, patience=2) |

---

## 📉 Results

| Metric   | Training | Validation |
| :------- | :------: | :--------: |
| Loss     |   0.283  |    0.332   |
| Accuracy |   ~91%   |   ~88–90%  |

Training stopped automatically after ~30 epochs using early stopping.

### 📊 Learning Curves

* **Train loss** decreased smoothly
* **Validation loss** plateaued early (no overfitting observed)
* Scheduler reduced LR dynamically when progress slowed

---

## 🖼️ Visualization

Predictions from validation set (40 random samples):

```python
plt.figure(figsize=(20,20))
for i in range(40):
    img = images[i].cpu().permute(1,2,0).numpy()
    img = (img * 0.5) + 0.5  # unnormalize
    label_text = "Dog 🐶" if preds[i] == 1 else "Cat 🐱"
    plt.subplot(5, 8, i + 1)
    plt.imshow(img)
    plt.title(label_text)
    plt.axis('off')
plt.show()
```

🧾 Each image shows the **predicted label** (`Cat` or `Dog`) under the image.

---

## 🧩 Project Structure

```
cat-and-dog-cnn-pytorch/
│
├── cat-and-dog-cnn-pytorch.ipynb   # Main training notebook
├── data/                           # Dataset folder (train/test)
├── best_model.pth                  # Saved best model
├── README.md                       # Project documentation
└── outputs/                        # Visualizations and logs
```

---

## 💡 Future Improvements

* Implement **Transfer Learning** using ResNet18 or EfficientNet
* Add **Grad-CAM** to visualize attention maps
* Deploy the model via **Gradio** or **Streamlit**
* Use a **larger dataset** for better generalization

---

## 🏆 Author

**Huỳnh Bảo Điền**
🎓 Computer Science
📫 Contact: Huynhdien2703@gmail.com

> “Building strong foundations in AI through practical projects.”

---
