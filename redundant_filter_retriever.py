from langchain_community.vectorstores.chroma import Chroma
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

class RedundantFilterRetriever(BaseRetriever):
    embeddings: Embeddings
    db: Chroma

    def _get_relevant_documents(self, query):
        embedding = self.embeddings.embed_query(query)
        return self.db.max_marginal_relevance_search_by_vector(
            embedding=embedding,
            lambda_mult=0.8,  # Multiplier for the diversity factor
        )

    async def _aget_relevant_documents(self):
        return []
