irs = {
    'task_metadata': {
        'task_id': 'economic_conditions_clustering',
        'task_type': 'spatial_clustering',
        'user_question': 'Using Census tract level data, which contains basic indicators from American Community Survey (ACS) data—such as poverty rate (B17020_cal), median family income (Tract_Fami), and income thresholds like State_Fa_1 or Metro_Fa_1—consider the following: Which two or three indicators would you choose to define broad economic conditions across communities? What are a few simple types of economic profiles you might expect to identify?',
        'Package Choice': ['geopandas', 'scikit-learn']
    },
    'data_description': {
        'data_type': 'polygon',
        'data_source': '/home/zl22853/code/ai_agent/data/census_tracts_south/Census_Tracts_South.shp',
        'spatial_reference': 'EPSG:4326',
        'variables': {
            'GEOID': {'description': 'Unique identifier for geographic areas', 'type': 'object'},
            'B17020_cal': {'description': 'Calculated value related to poverty statistics', 'type': 'float64'},
            'Tract_Fami': {'description': 'Total number of families within the census tract', 'type': 'int64'},
            'State_Fa_1': {'description': 'Specific count related to families within the state', 'type': 'int64'},
            'Metro_Fa_1': {'description': 'Specific count related to families within the metropolitan area', 'type': 'int64'}
        }
    },
    'analysis_goal': {
        'description': 'Identify broad economic conditions across communities using selected indicators.'
    },
    'data_preprocessing': {
        'handle_missing_values': 'def handle_missing_values(data):\n    # Impute mean for numerical variables and mode for categorical variables\n    data.fillna(data.mean(), inplace=True)\n    return data',
        'normalize_data': '["B17020_cal", "Tract_Fami", "State_Fa_1", "Metro_Fa_1"]',
        'remove_multi-collinearity': '["B17020_cal", "Tract_Fami", "State_Fa_1", "Metro_Fa_1"]'
    },
    'clustering_pipeline': [
        {
            'step_id': 1,
            'dep': -1,
            'step_name': 'Normalize Data',
            'method': 'normalize_data',
            'parameters': {
                'target_dimensions': ['B17020_cal', 'Tract_Fami', 'State_Fa_1', 'Metro_Fa_1'],
                'additional_parameters': {}
            }
        },
        {
            'step_id': 2,
            'dep': 1,
            'step_name': 'Remove Multicollinearity',
            'method': 'remove_multicollinearity',
            'parameters': {
                'target_dimensions': ['B17020_cal', 'Tract_Fami', 'State_Fa_1', 'Metro_Fa_1'],
                'additional_parameters': {}
            }
        },
        {
            'step_id': 3,
            'dep': 2,
            'step_name': 'Cluster Analysis',
            'method': 'KMeans',
            'parameters': {
                'target_dimensions': ['B17020_cal', 'Tract_Fami', 'State_Fa_1', 'Metro_Fa_1'],
                'additional_parameters': {'n_clusters': 3}
            }
        }
    ],
    'evaluation_plan': {
        'cluster_validity_metrics': ['silhouette_score', 'davies_bouldin_score'],
        'evaluation_strategy': {
            'variable_importance_analysis': ['canonical_correlation_analysis'],
            'visualization_methods': ['scatter_plot', 'heatmap']
        }
    }
}