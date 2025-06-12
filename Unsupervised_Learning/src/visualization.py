from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def pca_visualization(X_scaled, labels):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X_scaled)
    df = pd.DataFrame(reduced, columns=['PC1', 'PC2'])
    df['Cluster'] = labels
    sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='Set2')
    plt.title('Customer Segments via PCA')
    plt.show()
    
def analyze_clusters(original_df, labels):
    df = original_df.copy()
    df['Cluster'] = labels
    return df.groupby('Cluster').median()
