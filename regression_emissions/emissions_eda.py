import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(filepath):
    return pd.read_csv(filepath)

def plot_feature_distributions(df):
    df.hist(figsize=(12, 10))
    plt.tight_layout()
    plt.show()

def plot_scatter_matrix(df):
    sns.pairplot(df)
    plt.show()

def plot_correlation_heatmap(df):
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()
