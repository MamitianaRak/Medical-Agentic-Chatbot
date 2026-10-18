from functools import lru_cache
from pinecone import Pinecone, ServerlessSpec
from app.config import settings

@lru_cache()
def clients():
    if not settings.pinecone_api_key:
        raise ValueError("Pinecone API key must be set in the environment variables.")
    return Pinecone(api_key=settings.pinecone_api_key)


def ensure_index():
    pc = clients()
    if not pc.has_index(settings.pinecone_index):
        pc.create_index(name=settings.pinecone_index, dimension=settings.embed_dimensions, metric='cosine', spec=ServerlessSpec(cloud='aws', region='us-east-1'))
    else:
        dimension = pc.describe_index(settings.pinecone_index).dimension
        if dimension != settings.embed_dimensions:
            raise ValueError(f"Pinecone index '{settings.pinecone_index}' has {dimension} dimensions but embeddings use {settings.embed_dimensions}. Use a new index name.")
    return pc.Index(settings.pinecone_index)


def embed(texts, input_type='passage'):
    # Pinecone Inference: 'passage' for indexed documents, 'query' for user questions
    result = clients().inference.embed(model=settings.embed_model, inputs=texts, parameters={'input_type': input_type, 'truncate': 'END', 'dimension': settings.embed_dimensions})
    return [e.values for e in result.data]



def search_kb(query, top_k=5):
    index = ensure_index()
    vector = embed([query], input_type='query')[0]
    result = index.query(vector=vector, top_k=top_k, include_metadata=True, namespace=settings.pinecone_namespace)
    return [{'page': int(m.metadata.get('page', 0)), 'text': m.metadata.get('text',''), 'score': round(m.score,3), 'source': 'Gale Encyclopedia (2002)'} for m in result.matches if m.score >= 0.30]
