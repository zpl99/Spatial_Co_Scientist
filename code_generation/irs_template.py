irs_template = {
    "task_metadata": {
        "task_id": "unique_task_identifier",
        "task_type": "spatial_clustering",
        "user_question": "natural language question from user"
    },

    "data_description": {
        "data_type": "polygon or polyline or point or tabular",  # "polygon" | "polyline" | "point" | "tabular"
        "data_source": "path or url to dataset",
        "spatial_reference": "e.g., EPSG:4326",
        "variables": [
            {
                "name": "variable_name",
                "description": "brief description",
                "type": "",
            }
        ]
    },

    "analysis_goal": {
        "description": "brief textual description of the goal"
    },

    "clustering_pipeline": ["The following are some examples",
        {
            "step_id": 1,
            "dep": [-1],
            "step_name": "dimensionality_reduction",
            "method": {},  # or "t-SNE", "UMAP", "none"
            "parameters": {
                "target_dimensions": {},
                "additional_parameters": {}
            }
        },
        {
            "step_id": 2,
            "dep": [1],
            "step_name": "initial_clustering",
            "method": "k-means",  # or "DBSCAN", "HDBSCAN", etc.
            "parameters": {
                "number_of_clusters": "auto",
                "distance_metric": "euclidean",
                "spatial_constraint": "none",
                "additional_parameters": {}
            }
        },
        {
            "step_id": 3,
            "dep": [2],
            "step_name": "cluster_refinement",
            "method": "none",  # or "spatially_constrained_clustering"
            "parameters": {
                "constraint_type": "contiguity",
                "additional_parameters": {}
            }
        }
    ],

    "data_preprocessing": {
        "handle_missing_values": "impute_mean",
        "normalize_data": [],
        "remove_multicollinearity":[]
    },

    "evaluation_plan": {
        "number_of_iterations": [],
        "cluster_validity_metrics": [],
        "evaluation_strategy": {
            "visual_inspection": [],
            "domain_knowledge_check": [],
            "statistical_significance_check": [],
            "variable_importance_analysis": ["boxplot", "canonical_correlation_analysis"]
        }
    },
}