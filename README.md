# 🧠 Fetal Brain Anomaly Detection Using Deep Learning

This project aims to detect anomalies in fetal brain MRI images using deep learning techniques. It provides a web-based interface where users can upload MRI images, select a classification model, and receive detailed predictions along with a downloadable report.

---

## 📌 Features

- 📷 Upload fetal brain MRI images
- 🤖 Classify images as Normal or Abnormal using trained deep learning models
- 📄 Generate and download medical reports
- 🔐 Secure user authentication (Sign up / Login)
- 🧠 Select from multiple CNN-based models (e.g., Xception, ResNet)
- ⚠️ Detect and reject invalid or non-MRI images
- 📊 Display confidence scores and results

---

## 🛠️ Tech Stack

| Component      | Technology              |
|----------------|--------------------------|
| Frontend       | HTML5, CSS3              |
| Backend        | Python, Flask            |
| Deep Learning  | TensorFlow/Keras         |
| Image Handling | OpenCV, Pillow           |
| Authentication | Firebase                 |


---

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/appu-ui/fetal-brain-health-classification.git
   cd fetal-brain-anomaly-detection
2. Create a virtual environment and install dependencies:
    ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
3.Run the Flask application:
```bash
  python app.py
 






