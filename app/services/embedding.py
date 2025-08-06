import openai
from typing import List, Optional
import numpy as np
import os
import faiss

class EmbeddingModel:
    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        api_key: Optional[str] = None
    ):

        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY must be set as an environment variable or passed in."
                )
        openai.api_key = api_key
        self.model_name = model_name

    def embed(self, texts: List[str]) -> np.ndarray:

        response = openai.embeddings.create(
            model=self.model_name,
            input=texts
        )
        # response.data is a list of dicts, each with an 'embedding' list
        embeddings = [item.embedding for item in response.data]
        return np.vstack(embeddings)

def build_faiss_index(text_chunks, embed_model):
    # 1) Embed all chunks first
    embeddings = embed_model.embed(text_chunks)
    # 2) Figure out the dimensionality from the returned array
    #    embeddings.shape == (n_chunks, embedding_dim)
    n_chunks, dim = embeddings.shape

    # 3) Create the index with the correct dimension
    index = faiss.IndexFlatL2(dim)

    # 4) Add your embeddings (cast to float32 if needed)
    index.add(embeddings.astype(np.float32))

    return index, text_chunks


def search_faiss(query, index, text_chunks, embed_model, top_k=4):
    query_embedding = embed_model.embed([query])
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
    results = [text_chunks[idx] for idx in indices[0]]
    return results