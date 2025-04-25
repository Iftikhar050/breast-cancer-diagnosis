
# 🧠 Breast Cancer Prediction using Machine Learning

This project uses various machine learning algorithms to predict whether a tumor is **malignant** or **benign** based on the features from a breast cancer dataset.

---

## 📊 Dataset

- The dataset used is the **Breast Cancer Wisconsin (Diagnostic) Dataset** from the UCI Machine Learning Repository.
- It includes features computed from digitized images of fine needle aspirate (FNA) of breast masses.

---

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- Pandas, NumPy
- Seaborn & Matplotlib (for visualization)
- Scikit-learn (for ML models)

---

## 🔍 Features Selected

Features with a correlation > 0.5 to the diagnosis target were selected using a correlation heatmap and analysis.

---

## 🤖 Models Trained

The following models were implemented and evaluated:

| Model              | Accuracy (%) |
|-------------------|--------------|
| Logistic Regression | 98.2        |
| Decision Tree       | ~97         |
| Random Forest       | ~97         |
| SVM (Support Vector Machine) | ~97 |
| KNN (K-Nearest Neighbors)    | ~95 |

- Logistic Regression achieved the highest accuracy and balanced performance on precision and recall.

---

## 📈 Results

- **Accuracy Achieved:** 98.2%
- **Balanced performance** on both classes
- Models generalize well due to balanced dataset and train-test split

---

## 🧪 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/Iftikhar050/breast-cancer-diagnosis.git
   cd breast-cancer-diagnosis
   ```

2. Open the notebook:
   - Launch Jupyter Notebook and open `Breast_cancer_model.ipynb`

3. Run all cells:
   - The notebook will preprocess data, train models, and output results.

---

## 💾 Model Saving

- Trained models can be saved using `joblib` or `pickle` for deployment.
- Example:
  ```python
  import joblib
  joblib.dump(model, 'model.pkl')
  ```

---

## 📌 Project Status

✅ Completed – All models implemented and evaluated.  
📤 Ready for deployment or further feature engineering.

---

## 🙋‍♂️ Author

- [@Iftikhar050](https://github.com/Iftikhar050)

---

## 📝 License

This project is open-source and free to use under the [MIT License](LICENSE).
