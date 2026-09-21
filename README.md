# 🍛 Deep Learning — Padang Food Classification

A Deep Learning project for **classifying traditional Padang food images** using the **Xception architecture with Transfer Learning**.

The trained model is implemented into an interactive web application using **Streamlit**, allowing users to upload food images and receive classification results.

## 🚀 Features

* 🧠 **Xception Transfer Learning** — Uses the Xception architecture with Transfer Learning for image classification.
* 🍛 **Padang Food Classification** — Classifies images into multiple categories of traditional Padang food.
* 🖼️ **Image Classification** — Accepts food images as input for prediction.
* 📊 **Model Evaluation** — Evaluates model performance and generates an evaluation report.
* 🌐 **Streamlit Web App** — Provides an interactive web interface for using the trained model.
* 🧪 **Model Testing** — Includes testing functionality for validating model predictions.

## 🛠️ Tech Stack

| Technology         | Purpose                           |
| ------------------ | --------------------------------- |
| Python             | Programming language              |
| TensorFlow / Keras | Deep Learning framework           |
| Xception           | Image classification architecture |
| Transfer Learning  | Model training approach           |
| Streamlit          | Interactive web application       |
| NumPy              | Numerical computation             |
| Matplotlib         | Data visualization                |
| Git                | Version control                   |

## 📂 Project Structure

```text
Deep-Learning/
├── dataset_padang_food/     # Dataset for Padang food classification
├── results/                 # Model and evaluation results
│   ├── best_model.keras     # Trained model
│   └── evaluation_report.txt
├── model_training.py        # Model training and evaluation
├── app.py                   # Streamlit web application
├── test_model.py            # Model testing
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## ⚙️ Getting Started

### Prerequisites

Make sure you have the following installed:

* Python 3.x
* Git

It is recommended to use a **virtual environment** before installing the project dependencies.

### 1. Clone the Repository

```bash
git clone https://github.com/shendytria/Deep-Learning.git
cd Deep-Learning
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment.

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🧠 Model Training

To train the Xception Transfer Learning model and generate the trained model together with the evaluation report, run:

```bash
python model_training.py
```

The trained model and evaluation results will be stored in the `results/` directory.

## 🧪 Model Testing

To test the trained model, run:

```bash
python test_model.py
```

The testing script can be used to evaluate the model's prediction performance on the provided test data.

## 🌐 Running the Streamlit Application

After the trained model is available, run the Streamlit application:

```bash
streamlit run app.py
```

The application will be available through the local Streamlit server, typically at:

```text
http://localhost:8501
```

Users can upload a Padang food image through the web interface and receive the model's predicted classification.

## 🧠 Transfer Learning

This project uses **Transfer Learning** with the **Xception** architecture.

Instead of training an image classification model entirely from scratch, the project utilizes an existing pretrained model and adapts it to the Padang food classification task.

The general workflow is:

```text
Food Image
    │
    ▼
Image Preprocessing
    │
    ▼
Xception Transfer Learning
    │
    ▼
Model Training
    │
    ▼
Model Evaluation
    │
    ▼
Trained Model
    │
    ▼
Streamlit Application
    │
    ▼
Food Classification Result
```

## 📊 Model Results

The trained model and evaluation report are available in:

```text
results/
├── best_model.keras
└── evaluation_report.txt
```

The evaluation report contains the performance results obtained from the trained model.

## 📌 Project Purpose

This project was developed to apply **Deep Learning and Transfer Learning** concepts to an image classification problem involving traditional Padang food.

Through this project, several concepts were implemented, including:

* Image classification
* Convolutional Neural Networks
* Transfer Learning
* Xception architecture
* Model training and evaluation
* Model testing
* Deep Learning model deployment
* Interactive web application development using Streamlit

## 👩🏻‍💻 Developer

**Shendy Tria Amelyana**

D4 Teknik Informatika — Universitas Airlangga
