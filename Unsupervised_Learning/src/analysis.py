def analyze_clusters(original_df, labels):
    df = original_df.copy()
    df['Cluster'] = labels
    return df.groupby('Cluster').median()
