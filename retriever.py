"""
RAG Retriever Module - Handles context retrieval from vector store
"""
import logging
from typing import Tuple, List, Optional
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import RetrieverConfig


class RAGRetriever:
    """Handles document retrieval from vector store."""
    
    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.embeddings = None
        self.vectorstore = None
        
    def validate_vector_store(self) -> bool:
        vector_path = Path(self.config.vector_db_path)
        
        if not vector_path.exists():
            self.logger.error(f"Vector store not found: {self.config.vector_db_path}")
            self.logger.info("Please run the ingestion script first to create the vector store")
            return False
            
        if not vector_path.is_dir():
            self.logger.error(f"Path is not a directory: {self.config.vector_db_path}")
            return False
        
        index_file = vector_path / "index.faiss"
        pkl_file = vector_path / "index.pkl"
        
        if not index_file.exists():
            self.logger.error(f"FAISS index file not found: {index_file}")
            return False
            
        if not pkl_file.exists():
            self.logger.error(f"FAISS pickle file not found: {pkl_file}")
            return False
            
        self.logger.info(f"Vector store validated: {self.config.vector_db_path}")
        return True
    
    def load_embeddings(self):
        try:
            if self.embeddings is None:
                self.logger.info(f"Loading embedding model: {self.config.embedding_model}")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.config.embedding_model,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                self.logger.info("Embedding model loaded successfully")
            return self.embeddings
        except Exception as e:
            self.logger.error(f"Error loading embeddings: {str(e)}")
            raise
    
    def load_vectorstore(self) -> Optional[FAISS]:
        try:
            if self.vectorstore is None:
                self.logger.info(f"Loading vector store from: {self.config.vector_db_path}")
                
                if not self.validate_vector_store():
                    return None
                
                embeddings = self.load_embeddings()
                
                self.vectorstore = FAISS.load_local(
                    self.config.vector_db_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                self.logger.info("Vector store loaded successfully")
                
            return self.vectorstore
        except Exception as e:
            self.logger.error(f"Error loading vector store: {str(e)}")
            raise
    
    def calculate_confidence(self, num_docs: int, k: int) -> float:
        if num_docs == 0:
            return 0.0
        
        confidence = self.config.confidence_base + (num_docs / k) * self.config.confidence_multiplier
        confidence = min(self.config.confidence_max, confidence)
        
        return round(confidence, 2)
    
    def extract_pages(self, docs) -> List[int]:
        pages = []
        
        for doc in docs:
            if "page" in doc.metadata:
                page_num = doc.metadata["page"] + 1
                pages.append(page_num)
        
        ranked_pages = list(dict.fromkeys(pages))
        return ranked_pages
    
    def retrieve_context(self, query: str, k: Optional[int] = None) -> Tuple[str, List[int], float]:
        if k is None:
            k = self.config.top_k
        
        try:
            self.logger.info(f"Retrieving context for query: '{query[:100]}...'")
            
            vectorstore = self.load_vectorstore()
            
            if vectorstore is None:
                self.logger.error("Failed to load vector store")
                return "", [], 0.0
            
            docs = vectorstore.similarity_search(query, k=k)
            
            if not docs:
                self.logger.warning("No documents retrieved for query")
                return "", [], 0.0
            
            self.logger.info(f"Retrieved {len(docs)} documents")
            
            context_chunks = []
            for i, doc in enumerate(docs):
                chunk_preview = doc.page_content[:100].replace('\n', ' ')
                self.logger.debug(f"Chunk {i+1}: {chunk_preview}...")
                context_chunks.append(doc.page_content)
            
            ranked_pages = self.extract_pages(docs)
            
            if ranked_pages:
                self.logger.info(f"Relevant pages: {ranked_pages}")
            else:
                self.logger.warning("No page metadata found in retrieved documents")
            
            confidence = self.calculate_confidence(len(docs), k)
            self.logger.info(f"Confidence score: {confidence}")
            
            context_text = "\n\n".join(context_chunks)
            
            return context_text, ranked_pages, confidence
        except Exception as e:
            self.logger.error(f"Error during retrieval: {str(e)}")
            return "", [], 0.0