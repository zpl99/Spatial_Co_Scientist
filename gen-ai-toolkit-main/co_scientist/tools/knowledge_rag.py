from nano_graphrag import GraphRAG, QueryParam

def get_knowledge_rag(query:str, rag_database_path:str = "/Users/zepingliu/Library/CloudStorage/OneDrive-TheUniversityofTexasatAustin/博士学习/6-Job/ESRI/Spatial_Co_Scientist/co_scientist/rag_database", use_azure_openai=True):
    """
    Get knowledge from RAG database based on the query.
    :param query: The query string to search in the RAG database.
    :param rag_database_path: Path to the RAG database.
    :param use_azure_openai: Whether to use Azure OpenAI for querying.
    :return: The context retrieved from the RAG database.
    """
    graph_func = GraphRAG(working_dir=rag_database_path, using_azure_openai=use_azure_openai)

    # Query the RAG database
    context = graph_func.query(query, param=QueryParam(mode="local", only_need_context=True))

    return context


