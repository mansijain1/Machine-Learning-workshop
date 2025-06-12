from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.clustering import elbow_method, plot_elbow, train_kmeans
from src.visualization import pca_visualization
from src.analysis import analyze_clusters

def main():
    df = load_data("/home/nashtech/Documents/Machine-Learning/data/Wholesale_customers_data.csv")
    X_scaled, X_original = preprocess_data(df)

    # Show Elbow Plot
    wcss = elbow_method(X_scaled)
    plot_elbow(wcss)

    print("DEBUG: About to ask for number of clusters...")
    # Let user choose k after visualizing elbow plot
    k = int(input("Enter the number of clusters (k) based on the Elbow plot: "))

    # Train and Evaluate
    kmeans, labels, score = train_kmeans(X_scaled, k)
    print(f"Silhouette Score: {score:.2f}")

    # Visualize and Analyze
    pca_visualization(X_scaled, labels)
    cluster_summary = analyze_clusters(X_original, labels)
    print("\nCluster Spending Summary:")
    print(cluster_summary)

if __name__ == "__main__":
    main()
