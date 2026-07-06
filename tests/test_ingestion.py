import pytest
from backend.ingestion.chunker import DocumentChunker

def test_chunker_basic_splitting():
    chunker = DocumentChunker()
    markdown_text = """# Section 1
This is the text of section 1. It is quite simple.

## Subsection 1.1
This is the subsection text.

# Section 2
Final section content.
"""
    chunks = chunker.chunk_markdown(markdown_text)
    
    # We expect 3 distinct chunks based on headings
    assert len(chunks) == 3
    
    assert chunks[0]["heading_path"] == "Section 1"
    assert chunks[0]["content"] == "This is the text of section 1. It is quite simple."
    
    assert chunks[1]["heading_path"] == "Section 1 > Subsection 1.1"
    assert chunks[1]["content"] == "This is the subsection text."
    
    assert chunks[2]["heading_path"] == "Section 2"
    assert chunks[2]["content"] == "Final section content."

def test_chunker_sliding_window():
    chunker = DocumentChunker()
    # Force max tokens to be very small for test purposes
    chunker.max_tokens = 10 
    
    text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"
    markdown_text = f"# Long Section\n{text}"
    
    chunks = chunker.chunk_markdown(markdown_text)
    
    # Text length is 15 words. At max_tokens=10 (~7 words per token limit), it should split into multiple chunks
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["heading_path"] == "Long Section"
