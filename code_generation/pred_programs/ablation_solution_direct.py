import geopandas as gpd
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the data
data_path = '/home/zl22853/code/ai_agent/data/census_tracts_south/Census_Tracts_South.shp'
gdf = gpd.read_file(data_path)

# Select relevant indicators for economic conditions
# Based on the description, we choose poverty rate (B17020_cal), median family income (Tract_Fami), and income thresholds (State_Fa_1, Metro_Fa_1)
selected_columns = ['B17020_cal', 'Tract_Fami', 'State_Fa_1', 'Metro_Fa_1']
df = gdf[selected_columns]

# Preprocessing: Handle missing values by filling them with the median of each column
df.fillna(df.median(), inplace=True)

# Normalize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Clustering analysis using KMeans
# We choose 3 clusters to identify broad economic profiles
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Visualize the clustering results
plt.figure(figsize=(10, 6))
for cluster in range(3):
    plt.scatter(df[df['Cluster'] == cluster]['B17020_cal'], df[df['Cluster'] == cluster]['Tract_Fami'], label=f'Cluster {cluster}')
plt.xlabel('Poverty Rate (B17020_cal)')
plt.ylabel('Median Family Income (Tract_Fami)')
plt.title('Economic Profiles Clustering')
plt.legend()
plt.show()

# Interpretation: Print the cluster centers to understand the economic profiles
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
print("Cluster Centers (Original Scale):")
for i, center in enumerate(cluster_centers):
    print(f"Cluster {i}: Poverty Rate = {center[0]:.2f}, Median Family Income = {center[1]:.2f}, State Income Threshold = {center[2]:.2f}, Metro Income Threshold = {center[3]:.2f}")