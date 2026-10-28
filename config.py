PIPELINE_CONFIG = {
    "version": "1.2.0",
    "ingestion": {
        "parser": "unstructured==0.10.0",
        "supported_formats": ["pdf", "docx", "html", "md"],
    },
    "chunking": {
        "strategy": "recursive",
        "chunk_size": 500,
        "chunk_overlap": 50,
    },
    "embedding": {
        "model": "text-embedding-3-small",
        "dimensions": 1536,
        "provider": "openai",
    },
    "retrieval": {
        "top_k": 5,
        "similarity_metric": "cosine",
        "min_score_threshold": 0.7,
    },
    "generation": {
        "model": "claude-sonnet-4-6",
        "max_tokens": 500,
        "temperature": 0.0,
    },
}
