# Streamlit Application for Machine Learning

An interactive web-based Machine Learning platform designed to facilitate data exploration, model training, evaluation, and real-time inference.

## Project Description
This application provides a user-friendly interface to upload custom datasets, perform quick exploratory data analysis (including descriptive statistics and feature histograms), select target variables, configure preprocessing options (like scaling), train multiple Machine Learning models (such as Random Forest, SVM, Logistic/Linear Regression, and Decision Trees), compare their performance using visual charts, and perform interactive predictions using sliders.

## Technologies & Core Libraries
- **Language**: Python
- **Web Interface**: Streamlit
- **Machine Learning**: Scikit-Learn (Logistic Regression, Linear Regression, SVM, SVR, Random Forests, Decision Trees)
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn


## Creating the virtual environment

If you have more than one version of Python installed on your machine, type the command below to force the installation of the venv development environment in version 3.11:

```python
py -3.11 -m venv venv
```

## Activating the virtual environment

```python
.\venv\Scripts\activate
```

## Upgrading pip

```python
python.exe -m pip install --upgrade pip
```

## Installation

```python
pip install -r requirements.txt
```

## Running the application

```python
streamlit run app.py
```

## Link to the GitHub repository

[Project 2 - Machine Learning](https://github.com/lucas-ladeira/Projet-2-MachineLearning)

## Link to the Kaggle Dataset

Data used for testing:
- [California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices)

- [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)