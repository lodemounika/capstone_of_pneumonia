# 🏥 AI Healthcare Decision Support System

An AI-powered web application that predicts:

* 🧪 Diabetes Risk
* 🫁 Pneumonia / COVID-19 from Chest X-rays

This system assists in early disease detection and supports better healthcare decision-making.

---

## 🚀 Features

* 🔍 Diabetes risk prediction using clinical health parameters
* 🫁 Chest X-ray classification (COVID / Pneumonia / Normal)
* 📊 Displays prediction confidence score
* 🌐 User-friendly interface built with Streamlit

---

## 🧠 Model Explanation

### 🔹 Diabetes Prediction

* Algorithm: **Random Forest Classifier**
* Input Features:

  * Glucose
  * BMI
  * Age
  * Insulin
  * Blood Pressure
  * Skin Thickness
  * Diabetes Pedigree Function
* Output:

  * Risk Level (Low / Moderate / High)
  * Probability Score

---

### 🔹 Chest X-ray Detection

* Model: **MobileNetV2 (Transfer Learning)**
* Approach:

  * Pretrained on ImageNet
  * Fine-tuned on medical X-ray dataset
* Classes:

  * COVID-19
  * Pneumonia
  * Normal
* Output:

  * Predicted Disease
  * Confidence Percentage

---

## 📂 Dataset

Chest X-ray dataset used:

🔗 https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

Includes:

* COVID images
* Pneumonia images
* Normal images

---

## 🖥️ Screenshots

### 🔹 Diabetes Prediction


[<img width="1800" height="1053" alt="diabetes png" src="https://github.com/user-attachments/assets/088688e8-a076-4a4e-afc1-eae75a65dcef" />
](screenshots/diabetes.png)

---

### 🔹 X-ray Detection

[<img width="1777" height="775" alt="chest_xray png" src="https://github.com/user-attachments/assets/8bca6db9-abce-4ba1-85e6-23a44fe9f960" />
](screenshots/chest_xray.png)

---

## ⚙️ Installation & Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
📦 AI-Healthcare-System
 ┣ 📜 app.py
 ┣ 📜 cnn.py
 ┣ 📜 diabetics.py
 ┣ 📜 chest_model.h5
 ┣ 📜 diabetes_rf_model.pkl
 ┣ 📜 scaler.pkl
 ┣ 📜 requirements.txt
 ┣ 📜 README.md
 ┗ 📁 screenshots/
```

---

## 📊 Results

* ✅ Diabetes Prediction Accuracy: ~80%+
* ✅ X-ray Classification Accuracy: ~85%+

---

## 🧪 How to Use

1. Open the web app
2. Select:

   * Diabetes Prediction → Enter values
   * X-ray Detection → Upload image
3. Click predict
4. View result and confidence score

---

## 🔮 Future Enhancements

* Integration with hospital systems
* Support for more diseases
* Improved accuracy using advanced models
* Cloud deployment for real-time access

---

## 🤝 Conclusion

This project demonstrates how Artificial Intelligence can be effectively used in healthcare to provide fast, reliable, and automated disease predictions, improving early diagnosis and decision-making.

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
