import logging
import re
from typing import List, Dict, Any, Optional
from backend.db.repository import RAGRepository
from backend.rag.embedder import Embedder

logger = logging.getLogger("app")

# Standard English stop words list for keyword scoring optimization
STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", 
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", 
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", 
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", 
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", 
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", 
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", 
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
}

class RetrievalEngine:
    """
    Search and retrieval coordinator. Integrates vector search queries 
    with a custom keyword overlap density scorer (linear rank fusion).
    Does NOT use heavy third-party framework wrappers (like LlamaIndex).
    """
    def __init__(self) -> None:
        self.embedder = Embedder()
        self.repository = RAGRepository()
        # Relative weights for semantic vs keyword matching (total = 1.0)
        self.semantic_weight = 0.7
        self.keyword_weight = 0.3

    def _tokenize(self, text: str) -> List[str]:
        """Cleans and splits text into tokens, removing standard stop words."""
        # Convert to lowercase and strip non-alphanumeric chars
        clean_text = re.sub(r"[^\w\s]", "", text.lower())
        tokens = clean_text.split()
        return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

    def calculate_keyword_score(self, query_tokens: List[str], chunk_content: str) -> float:
        """
        Calculates the keyword density match score between the query tokens and chunk text.
        Normalized to [0.0, 1.0].
        """
        if not query_tokens:
            return 0.0
            
        chunk_tokens = set(self._tokenize(chunk_content))
        if not chunk_tokens:
            return 0.0
            
        # Count overlapping tokens
        matches = sum(1 for token in query_tokens if token in chunk_tokens)
        return float(matches) / len(query_tokens)

    async def retrieve_relevant_chunks(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic vector retrieval against MongoDB, applies pre-filters,
        calculates custom keyword overlap scores, and re-ranks final candidates.
        """
        logger.info(f"Initiating semantic retrieval for query: '{query}'")
        
        # 1. Generate Query Vector Embedding
        query_vector = await self.embedder.get_embedding(query)
        
        # 2. Retrieve Candidate Chunks from MongoDB Vector Store
        # We retrieve double the required candidate count (Top-K * 4) to perform re-ranking
        candidate_limit = max(top_k * 4, 20)
        
        # Format filters to MongoDB query format if they exist
        mongo_filter = None
        if filters:
            mongo_filter = {}
            for key, val in filters.items():
                if val is not None:
                    mongo_filter[key] = val
                    
        candidates = await self.repository.vector_search(
            query_vector=query_vector,
            limit=candidate_limit,
            pre_filter=mongo_filter
        )
        
        if not candidates:
            logger.info("No matching candidates found in vector search.")
            return []
            
        logger.info(f"Retrieved {len(candidates)} candidates from DB. Executing hybrid re-ranking...")
        
        # 3. Apply Re-ranking and Score Fusion
        query_tokens = self._tokenize(query)
        re_ranked_candidates = []
        
        for candidate in candidates:
            # Vector Cosine Score (typically [0.0, 1.0] returned by Atlas)
            # Ensure it is bounded
            vector_score = float(candidate.get("score", 0.0))
            
            # Keyword Density Score
            keyword_score = self.calculate_keyword_score(query_tokens, candidate["content"])
            
            # Linear Weighted Fusion
            fused_score = (self.semantic_weight * vector_score) + (self.keyword_weight * keyword_score)
            
            # Append scores to candidate metadata
            candidate_copy = candidate.copy()
            candidate_copy["vector_score"] = vector_score
            candidate_copy["keyword_score"] = keyword_score
            candidate_copy["hybrid_score"] = fused_score
            
            re_ranked_candidates.append(candidate_copy)
            
        # Sort by fused score descending
        re_ranked_candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        # Take Top-K
        final_results = re_ranked_candidates[:top_k]
        logger.info(f"Re-ranking complete. Returning Top-{len(final_results)} chunks to generative layer.")
        
        return final_results
