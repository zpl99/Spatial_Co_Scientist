import os
from nano_graphrag import GraphRAG, QueryParam


graph_func = GraphRAG(working_dir="/Users/zepingliu/Library/CloudStorage/OneDrive-TheUniversityofTexasatAustin/博士学习/6-Job/ESRI/Spatial_Co_Scientist/co_scientist/rag_database",using_azure_openai=True)


txt_dir = "/Users/zepingliu/Library/CloudStorage/OneDrive-TheUniversityofTexasatAustin/博士学习/6-Job/ESRI/Spatial_Co_Scientist/data/spatial_cluster_memory_txt"  # 替换为你的txt文件夹路径

# 遍历目录下所有txt文件
for filename in os.listdir(txt_dir):
    if filename.lower().endswith(".txt"):
        file_path = os.path.join(txt_dir, filename)
        print(f"Inserting file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            graph_func.insert(content)
