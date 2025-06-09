import json
from pathlib import Path

irs_options = {
    "data_type": ["polygon", "polyline", "point", "tabular"],
    "goal_type": [
        "homogeneous_regions",
        "density_hotspots",
        "multivariate_profiles",
        "resilience_profiles",
        "economic_profiles"
    ],
    "dimensionality_reduction_method": ["PCA", "t-SNE", "UMAP", "none"],
    "initial_clustering_method": ["k-means", "DBSCAN", "HDBSCAN", "hierarchical_clustering"],
    "distance_metric": ["euclidean", "manhattan", "geographic", "user_defined"],
    "spatial_constraint": ["none", "contiguity", "distance_band"],
    "cluster_refinement_method": ["spatially_constrained_clustering", "none"],
    "handle_missing_values": ["remove_rows", "impute_mean", "impute_median", "none"],
    "cluster_validity_metrics": ["pseudo_f", "silhouette_score", "davies_bouldin"],
    "variable_importance_analysis": ["boxplot", "canonical_correlation_analysis"],
    "preferred_spatial_scale": ["fine", "medium", "coarse"],
    "variable_type": ["numeric", "categorical", "ordinal"]
}

#
output_dir = Path("/Users/zepingliu/Library/CloudStorage/OneDrive-TheUniversityofTexasatAustin/博士学习/6-Job/ESRI/ai_agent")
output_dir.mkdir(parents=True, exist_ok=True)

for key, values in irs_options.items():
    file_path = output_dir / f"{key}.json"
    with file_path.open("w") as f:
        json.dump(values, f, indent=2)

