import os
import logging
from markitdown import MarkItDown
from backend.core.config import settings

logger = logging.getLogger("app")

class DocumentParser:
    """
    Parser to convert various document formats (PDF, DOCX, TXT, MD) 
    into structured Markdown using Microsoft's MarkItDown.
    """
    def __init__(self) -> None:
        self.md_converter = MarkItDown()
        # Initialize temp directory inside the workspace root
        self.workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.temp_dir = os.path.join(self.workspace_root, "data", "tmp")
        os.makedirs(self.temp_dir, exist_ok=True)

    def parse_file(self, file_bytes: bytes, filename: str) -> str:
        """
        Saves file bytes to a temporary directory in the workspace, 
        converts the document to Markdown, and cleans up the file.
        """
        temp_file_path = os.path.join(self.temp_dir, f"{os.getpid()}_{filename}")
        
        try:
            logger.info(f"Writing temporary file to {temp_file_path} for parsing")
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(file_bytes)
                
            logger.info(f"Converting document {filename} to Markdown using MarkItDown")
            result = self.md_converter.convert(temp_file_path)
            
            if not result or not result.text_content:
                raise ValueError(f"MarkItDown returned empty content for {filename}")
                
            logger.info(f"Successfully converted {filename} to Markdown ({len(result.text_content)} chars)")
            return result.text_content
            
        except Exception as e:
            logger.error(f"Failed to parse document {filename}: {str(e)}", exc_info=True)
            raise RuntimeError(f"Document parsing failed: {str(e)}") from e
            
        finally:
            # Enforce deletion of temp file
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.debug(f"Removed temporary file {temp_file_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Could not remove temp file {temp_file_path}: {str(cleanup_error)}")
