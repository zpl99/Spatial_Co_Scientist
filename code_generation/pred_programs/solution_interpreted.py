import geopandas as gpd
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data_source = '/home/zl22853/code/ai_agent/data/census_tracts_south/Census_Tracts_South.shp'
gdf = gpd.read_file(data_source)

# Handle missing values
def handle_missing_values(data):
    # Impute mean for numerical variables
    for column in data.select_dtypes(include=['float64', 'int64']).columns:
        data[column].fillna(data[column].mean(), inplace=True)
    return data

gdf = handle_missing_values(gdf)

# Normalize data
features = ["B17020_cal", "Tract_Fami", "State_Fa_1", "Metro_Fa_1"]
scaler = StandardScaler()
gdf[features] = scaler.fit_transform(gdf[features])

# Remove multicollinearity using PCA
pca = PCA(n_components=2)
gdf_pca = pca.fit_transform(gdf[features])

# Cluster Analysis using KMeans
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
gdf['cluster'] = kmeans.fit_predict(gdf_pca)

# Evaluate clustering
silhouette_avg = silhouette_score(gdf_pca, gdf['cluster'])
davies_bouldin_avg = davies_bouldin_score(gdf_pca, gdf['cluster'])

print(f'Silhouette Score: {silhouette_avg}')
print(f'Davies-Bouldin Score: {davies_bouldin_avg}')

# Print cluster sizes and centroids
print("Cluster Sizes:")
print(gdf['cluster'].value_counts())

print("Cluster Centroids:")
print(kmeans.cluster_centers_)

# Visualization
# Scatter plot of clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=gdf_pca[:, 0], y=gdf_pca[:, 1], hue=gdf['cluster'], palette='viridis')
plt.title('Cluster Scatter Plot')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()

# Map visualization
gdf.plot(column='cluster', cmap='viridis', legend=True, figsize=(12, 8))
plt.title('Geospatial Clustering Map')
plt.show()

# Heatmap of feature correlations
plt.figure(figsize=(8, 6))
sns.heatmap(gdf[features].corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()