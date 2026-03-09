# 🩸 HematoVision – Blood Cell Classification System

HematoVision is an AI-powered web application that detects and classifies different types of blood cells using deep learning. The system allows users to upload microscopic blood cell images and automatically predicts the type of cell using a trained neural network model.

This project combines **Machine Learning, Computer Vision, and Web Development** to assist in medical image analysis and hematology research.

---

# ✨ Features

🔬 **Blood Cell Detection** – Upload microscopic blood cell images for classification

🧠 **Deep Learning Model** – Uses a trained CNN model (`Blood Cell.h5`) for accurate prediction

🌐 **Web Interface** – Simple and interactive user interface built with HTML and Flask

📤 **Image Upload System** – Users can upload blood cell images easily

📊 **Prediction Results** – Displays the predicted blood cell type instantly

⚡ **Fast Processing** – Predictions generated in seconds

📱 **User-Friendly UI** – Clean web pages for easy navigation

---

# 🚀 Tech Stack

## Backend

* Python
* Flask
* TensorFlow / Keras
* NumPy
* OpenCV / PIL

## Frontend

* HTML5
* CSS
* Bootstrap (optional styling)

## Machine Learning

* Convolutional Neural Networks (CNN)
* Pre-trained deep learning model (`Blood Cell.h5`)

---

# 📋 Prerequisites

Before running the project, make sure you have:

* Python **3.8 or higher**
* pip installed
* Virtual environment (recommended)

---

# 🔧 Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Hematovision.git
cd Hematovision
```

---

## 2️⃣ Install Dependencies

Install all required libraries from `requirements.txt`.

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Project Structure

```
Hematovision/
│
├── app.py                 # Flask application
├── Blood Cell.h5          # Trained deep learning model
├── requirements.txt       # Python dependencies
│
├── templates/             # HTML templates
│   ├── Home.html
│   ├── Project.html
│   ├── result.html
│   └── team.html
│
└── README.md
```

---

# ▶️ Running the Application

Start the Flask server:

```bash
python app.py
```

The application will run at:

```
http://127.0.0.1:5000
```

Open this URL in your browser.

---

# 📖 Usage

### Step 1 – Open the Web App

Go to the homepage.

### Step 2 – Upload Image

Upload a microscopic image of a blood cell.

### Step 3 – Model Prediction

The deep learning model analyzes the image.

### Step 4 – View Results

The system displays the predicted **blood cell type**.

---

# 🧠 How the Model Works

1. User uploads a blood cell image.
2. The image is preprocessed (resized and normalized).
3. The trained CNN model (`Blood Cell.h5`) processes the image.
4. The model predicts the class of the blood cell.
5. The result is displayed on the result page.

---

# 📊 Blood Cell Types Detected

The model can classify different types of blood cells such as:

* 🟥 Red Blood Cells (RBC)
* ⚪ White Blood Cells (WBC)
* 🟣 Platelets
* Other cell variations depending on the training dataset.

---

# 🛠️ Future Enhancements

📈 Improve model accuracy with a larger dataset
📊 Add prediction confidence scores
📱 Build a mobile-friendly interface
📷 Enable real-time image capture via webcam
☁️ Deploy the system on cloud platforms (AWS / Heroku)
📊 Add visualization charts for analysis

---

# 👨‍💻 Team

This project was developed as part of an academic / research project.

Team members are listed on the **Team Page** of the web application.

---

# 📄 License

This project is licensed under the **MIT License**.

---

# ⭐ Support

If you like this project:

* ⭐ Star the repository
* 🍴 Fork the project
* 🛠️ Contribute improvements

---

# 📬 Contact

For questions or collaboration, feel free to reach out through GitHub.

---
