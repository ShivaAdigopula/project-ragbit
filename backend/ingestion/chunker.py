import re
import logging
from typing import List, Dict, Any, Generator
from backend.core.config import settings

logger = logging.getLogger("app")

class DocumentChunker:
    """
    Splits Markdown text into semantic chunks based on headings and token limits.
    Maintains header hierarchy paths as metadata.
    """
    def __init__(self) -> None:
        self.max_tokens = settings.MAX_CHUNK_SIZE_TOKENS
        self.overlap_percent = settings.CHUNK_OVERLAP_PERCENTAGE
        # Approximate nomic-embed-text tokens: 1 word ~ 1.3 tokens
        self.words_per_token = 1.3

    def estimate_tokens(self, text: str) -> int:
        """Estimates token count based on word metrics."""
        word_count = len(text.split())
        return int(word_count * self.words_per_token)

    def split_text_sliding_window(self, text: str) -> List[str]:
        """
        Splits text into sliding windows based on max token rules and overlap percentages.
        """
        words = text.split()
        if not words:
            return []
            
        max_words = int(self.max_tokens / self.words_per_token)
        overlap_words = int(max_words * self.overlap_percent)
        
        # Ensure we make progress even with tiny thresholds
        if max_words <= 0:
            max_words = 100
        if overlap_words >= max_words:
            overlap_words = max_words // 2
            
        chunks = []
        start_idx = 0
        
        while start_idx < len(words):
            end_idx = min(start_idx + max_words, len(words))
            chunk_words = words[start_idx:end_idx]
            chunks.append(" ".join(chunk_words))
            
            # Move index forward
            if end_idx == len(words):
                break
            start_idx = end_idx - overlap_words
            
        return chunks

    def chunk_markdown(self, markdown_text: str) -> List[Dict[str, Any]]:
        """
        Splits Markdown text into blocks based on headings, then subdivides blocks 
        using a sliding window if they exceed token limits.
        
        Returns a list of dictionaries containing:
        - content: str (the text content of the chunk)
        - heading_path: str (hierarchy breadcrumb, e.g. "Root > Introduction > Background")
        - token_count: int
        """
        lines = markdown_text.splitlines()
        chunks: List[Dict[str, Any]] = []
        
        # Header stack tracking [("Header Text", Level)]
        header_stack: List[tuple[str, int]] = []
        current_block: List[str] = []
        
        def get_heading_path() -> str:
            if not header_stack:
                return "Root"
            return " > ".join([h[0] for h in header_stack])
            
        def process_current_block():
            block_text = "\n".join(current_block).strip()
            if not block_text:
                return
                
            heading_path = get_heading_path()
            estimated_tokens = self.estimate_tokens(block_text)
            
            if estimated_tokens <= self.max_tokens:
                chunks.append({
                    "content": block_text,
                    "heading_path": heading_path,
                    "token_count": estimated_tokens
                })
            else:
                # Exceeds max tokens -> run sliding window split
                logger.info(f"Chunk exceeds token limit ({estimated_tokens} tokens). Splitting window under: {heading_path}")
                sub_texts = self.split_text_sliding_window(block_text)
                for sub_text in sub_texts:
                    chunks.append({
                        "content": sub_text,
                        "heading_path": heading_path,
                        "token_count": self.estimate_tokens(sub_text)
                    })
            current_block.clear()

        # Regular expression for matching headers: # Heading Text
        header_regex = re.compile(r"^(#{1,6})\s+(.+)$")
        
        for line in lines:
            header_match = header_regex.match(line)
            if header_match:
                # Flush the block accumulated before the new header
                process_current_block()
                
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                # Maintain heading stack based on levels
                while header_stack and header_stack[-1][1] >= level:
                    header_stack.pop()
                    
                header_stack.append((title, level))
            else:
                current_block.append(line)
                
        # Flush any trailing block
        process_current_block()
        
        logger.info(f"Chunking complete. Created {len(chunks)} chunks from document")
        return chunks
