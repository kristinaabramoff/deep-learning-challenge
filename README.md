
# Alphabet Soup Funding Predictor

## Overview of the Analysis
The purpose of this analysis is to build and evaluate a deep learning model to predict whether applicants for funding from **Alphabet Soup** will be successful in their ventures. The dataset includes over 34,000 records of organizations that have received funding, with features such as application type, organization classification, and requested funding amount used to make predictions.

The goal is to help Alphabet Soup better allocate funding to applicants with the highest probability of success by developing a binary classifier.

## Tools and Libraries
### Programming Language:
- **Python**

### Libraries:
- **TensorFlow**: For building and training the neural network.
- **Pandas**: For data manipulation and preprocessing.
- **Scikit-learn**: For data scaling and splitting the dataset.
- **Matplotlib**: For data visualization and plotting.

### Software:
- **Jupyter Notebook**: Used for interactive development and testing.
- **Git**: For version control.
- **Integrated Development Environment (IDE)**: (Specify if using PyCharm, VSCode, or any other IDE).

## Data Preprocessing
### Target Variable:
- **IS_SUCCESSFUL**: Binary variable indicating whether the funding was used successfully.

### Feature Variables:
- **APPLICATION_TYPE** (17 categories, binned)
- **AFFILIATION** (6 categories)
- **CLASSIFICATION** (71 categories, binned)
- **USE_CASE** (5 categories)
- **ORGANIZATION** (4 categories)
- **STATUS** (2 categories)
- **INCOME_AMT** (9 categories)
- **SPECIAL_CONSIDERATIONS** (2 categories)
- **ASK_AMT** (Continuous variable)

### Removed Variables:
- **EIN** and **NAME**: These columns were removed as they are not relevant to the prediction.

### Binning and Encoding:
- Infrequent categories were binned into an "Other" category.
- All categorical variables were converted to numeric using **one-hot encoding**.

## Model Training and Evaluation
### Neurons, Layers, and Activation Functions:
#### Initial Model:
- **80 neurons** in the input layer.
- **2 hidden layers** with **ReLU** activation.
- **Sigmoid** activation function in the output layer.

#### Optimized Model:
- **90, 30, and 20 neurons** across **3 hidden layers**.
- Used **ReLU** and **Tanh** activation functions in the hidden layers.
- **Sigmoid** activation function in the output layer.

### Performance:
#### Initial Model:
- **Loss**: 0.5751
- **Accuracy**: 72.45%

#### Optimized Model:
- **Loss**: 0.5531
- **Accuracy**: 73.58%

### Steps Taken:
1. **Data Cleaning**: Removed non-beneficial columns and binned infrequent categories.
2. **Model Optimization**: Added a third hidden layer and experimented with different activation functions (ReLU and Tanh) to improve performance.
3. **Hyperparameter Tuning**: Adjusted the number of neurons in each layer to find the optimal balance between underfitting and overfitting.

## Summary and Recommendations
The final optimized model achieved an accuracy of **73.58%**. Although the model performed well, further improvements could be made by experimenting with other machine learning models such as **Random Forest Classifier** or **Gradient Boosting Machines**. Additionally, implementing **hyperparameter tuning** techniques like **grid search** may help improve the model's performance.

For a detailed analysis, model evaluation, and insights on the optimization process, please refer to the **report** available in the repository.

## Installation and Usage Instructions
To clone the repository and run the project on your local machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/yourproject.git


jupyter notebook AlphabetSoupCharity.ipynb

