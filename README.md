# 📰 Fake News Detection using Machine Learning (NLP)

## 📌 Overview
Fake news on digital platforms can mislead people and influence public opinion.  
This project implements a **Fake News Detection system** using **Machine Learning and Natural Language Processing (NLP)** to classify news headlines as **REAL** or **FAKE**.

The focus of this project is to understand the **complete NLP pipeline** — from text preprocessing to model evaluation and visualization.

---

## 🎯 Problem Statement
To build a machine learning model that can automatically identify whether a given news headline is **real or fake** using textual features.

---

## 📊 Dataset
The dataset is sourced from **Kaggle (FakeNewsNet)**.

### Columns used:
- **title** → News headline text (input feature)
- **real** → Label  
  - `1` → REAL news  
  - `0` → FAKE news  

Only the **headline text** was used to keep the model simple, fast, and interpretable.

---

## 🧠 Approach
1. Loaded the dataset using **Pandas**
2. Selected news headlines as input features
3. Converted text data into numerical form using **TF-IDF Vectorization**
4. Split the data into training and testing sets
5. Trained a **Naive Bayes classifier** for text classification
6. Evaluated the model using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
7. Visualized results using a **Confusion Matrix**

---

## 🛠️ Technologies Used
- Python
- Pandas
- scikit-learn
- TF-IDF Vectorization (NLP)
- Matplotlib
- Seaborn

---

## 📈 Results
- Achieved an accuracy of approximately **82%**
- The model performs very well in identifying **real news**
- Fake news recall is comparatively lower due to **class imbalance**, reflecting real-world challenges in NLP classification tasks

The goal of this project is **learning and interpretability**, not over-optimizing accuracy.

---

## 📊 Visualization
- A **Confusion Matrix** was used to visualize classification performance
- Clearly highlights correct and incorrect predictions
- Helps analyze model bias toward real news

---

## ▶️ How to Run the Project

1. Install dependencies:
   pip install -r requirements.txt
2.Run the model:
  python src/model.py

## 📚 Key Learnings
- Text data must be transformed into numerical features before applying ML models
- TF-IDF helps capture the importance of words in documents
- Accuracy alone is not sufficient for classification problems
- Confusion matrices provide deeper insights into model performance
- Real-world datasets often suffer from class imbalance

## 🔮 Future Improvements
- Use full news articles instead of only headlines
- Handle class imbalance using resampling techniques
- Experiment with advanced classifiers like Logistic Regression or SVM
- Deploy the model as a web application

## Author
Rahul Teja Kotta
B.Tech CSE (Artificial Intelligence)

## License
This project is open-source and intended for educational and learning purposes.
