irs_template = {
    "task_metadata": {
        "task_id": "unique_task_identifier",
        "task_type": "spatial_clustering",
        "user_question": "natural language question from user",
        "Package Choice": "Here, you can define what kind of python package you want to use, such as arcpy, geopandas, pandas, scikit-learn, etc. Following the format ['package1', 'package2', ...]"
    },

    "data_description": {
        "data_type": "polygon or polyline or point or tabular",
        "data_source": "path or url to dataset",
        "spatial_reference": "e.g., EPSG:4326",
        "variables": "Here, please detailed all the attributes you think is needed in this task, including the name, description, and type of each variable. Following the format {'name:', 'description:', 'type:'}"
    },

    "analysis_goal": {
        "description": "brief textual description of the goal"
    },

    "data_preprocessing": {
        "handle_missing_values": "impute_mean, but be aware of different attribute typem for example, for categorical variables, you may want to use impute_mode instead of impute_mean. The code should be in the format def handle_missing_values(data):\n    # Your code here\n    return data",
        "normalize_data": 'read the data I provide with you, please specify the variables to be normalized and only normalize the data you want to use. The code should follow the format ["variable1", "variable2", ...], and also write the python code for normalizae the data, the python code should be in the format def normalize_data(data, variables):\n    # Your code here\n    return data',
        "remove_multi-collinearity": 'read the data I provide with you, please specify the variables to remove multi-collinearity, following the format ["variable1", "variable2", ...], and also write the python code for removing multi-collinearity, the python code should be in the format def remove_multicollinearity(data, variables):\n    # Your code here\n    return data'
    },

    "clustering_pipeline": "Here, please detailed all the steps in the clustering pipeline, including the step id, dependencies, step name, method, and parameters. Following the format [{'step_id': [the id of this step], 'dep': [the dependent id of the step, if this is no dependency, leave it to be -1], 'step_name': [], 'method': {}, 'parameters': {'target_dimensions': {}, 'additional_parameters': {}}}, ...]. ",

    "evaluation_plan": {
        "cluster_validity_metrics": "Here, please detailed the cluster validity metrics that is suitable for this case, following the format ['metric1', 'metric2', ...]",
        "evaluation_strategy": {
            "variable_importance_analysis": "The candidate choices are ['boxplot', 'canonical_correlation_analysis'].",
            "visualization_methods": "The candidate choices are ['scatter_plot', 'box_plot', 'heatmap']."
        }
    }
}