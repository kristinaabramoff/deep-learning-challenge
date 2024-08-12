
# Alphabet Soup Funding Predictor

## Overview of the Analysis
The purpose of this analysis is to build and evaluate a deep learning model to predict whether applicants for funding from Alphabet Soup will be successful. The dataset includes a variety of features that are used to train the model to identify patterns that correlate with successful outcomes.

## Tools and Libraries
- **Programming Language:** Python
- **Libraries:**
  - `TensorFlow` for building and training the neural network
  - `Pandas` for data manipulation and preprocessing
  - `Scikit-learn` for data scaling and splitting
  - `Matplotlib` for data visualization
- **Software:**
  - **Jupyter Notebook:** Used for interactive development and testing.
  - **Git:** For version control.
  - **Integrated Development Environment (IDE):** Mention if you used PyCharm, VSCode, or another IDE.

## Data Preprocessing
- **Target Variable:** `IS_SUCCESSFUL`
- **Feature Variables:**
  - `APPLICATION_TYPE` (17 categories, binned)
  - `AFFILIATION` (6 categories)
  - `CLASSIFICATION` (71 categories, binned)
  - `USE_CASE` (5 categories)
  - `ORGANIZATION` (4 categories)
  - `STATUS` (2 categories)
  - `INCOME_AMT` (9 categories)
  - `SPECIAL_CONSIDERATIONS` (2 categories)
  - `ASK_AMT` (Continuous variable)
- **Removed Variables:** `EIN` and `NAME`
- **Binning and Encoding:** Infrequent categories were binned, and all categorical variables were converted to numeric using one-hot encoding.

## Model Training and Evaluation
- **Neurons, Layers, and Activation Functions:**
  - Initial model: 80 neurons, 2 hidden layers with ReLU, output layer with Sigmoid
  - Optimized model: 90, 30, and 20 neurons across 3 hidden layers with ReLU and Tanh, output layer with Sigmoid
- **Performance:**
  - Initial model: Loss = 0.5751, Accuracy = 72.45%
  - Optimized model: Loss = 0.5531, Accuracy = 73.58%
- **Steps Taken:**
  - **Data Cleaning:** Removed non-beneficial columns and binned infrequent categories.
  - **Model Optimization:** Added a third hidden layer and experimented with activation functions to improve performance.
  - **Hyperparameter Tuning:** Adjusted the number of neurons in each layer to find the best balance between underfitting and overfitting.

## Summary and Recommendations
The model showed promising results with an accuracy of 73.58% after optimization. For further improvement, consider using a Random Forest Classifier or Gradient Boosting Machine, and implementing hyperparameter tuning.

## Installation and Usage Instructions
- Clone the repository:
  ```bash
  git clone https://github.com/yourusername/yourproject.git
