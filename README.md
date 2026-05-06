
# Iris Flower Classification
### CodeAlpha Data Science Internship - Task 1

## Project Overview
This project builds and evaluates machine learning models to classify
Iris flower species (Setosa, Versicolor, Virginica) based on sepal
and petal measurements.

## Dataset
- **Source:** Iris.csv (150 samples, 4 features, 3 classes)
- **Features:** SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
- **Target:** Species (Iris-setosa, Iris-versicolor, Iris-virginica)

## Workflow
1. Load Dataset
2. Clean Data (drop duplicates, handle missing values)
3. Exploratory Data Analysis (distributions, boxplots, heatmap, pairplot)
4. Preprocess (encode labels, train/test split 80/20)
5. Train Models
6. Evaluate & Compare

## Models Used
| Model | Accuracy |
|-------|----------|
| K-Nearest Neighbors | 100.00%  |
| Support Vector Machine | 96.67% |
| Decision Tree | 93.33% |
| Random Forest | 90.00% |

## Visualizations
- Feature distributions by species
- Boxplots per feature
- Correlation heatmap
- Pairplot
- Model accuracy comparison
- Confusion matrices

## Libraries Used
- Python 3.10
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## How to Run

### Run the Python script:
```bash
python3 iris_classification.py
```

### Or open the Jupyter Notebook:
```bash
jupyter notebook iris_classification.ipynb
```

## Files
| File | Description |
|------|-------------|
| `iris_classification.ipynb` | Jupyter Notebook — full analysis |
| `iris_classification.py` | Python script |
| `Iris.csv` | Dataset |
| `iris_distributions.png` | Feature distribution plots |
| `iris_boxplots.png` | Boxplots per feature |
| `iris_heatmap.png` | Correlation heatmap |
| `iris_pairplot.png` | Pairplot of all features |
| `iris_model_comparison.png` | Model accuracy bar chart |
| `iris_confusion_matrices.png` | Confusion matrices |

## Key Findings
- Petal Length and Petal Width are the strongest predictors of species
- Iris-setosa is perfectly separable from the other two species
- KNN achieved the best accuracy of 100% on the test set
- All 4 models performed above 90%

## Author
- **Internship:** CodeAlpha Data Science Internship
- **Task:** Task 1 - Iris Flower Classification
