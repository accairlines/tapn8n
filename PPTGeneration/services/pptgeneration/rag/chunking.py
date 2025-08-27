import re
import logging
from typing import List, Dict, Any
from django.conf import settings

logger = logging.getLogger(__name__)


class TextChunker:
    """Text chunking for email content."""
    
    def __init__(self, chunk_size=1000, overlap=150):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = 200  # Minimum chunk size
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Chunk text into overlapping segments."""
        if not text or len(text.strip()) == 0:
            return []
        
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Split into sentences first
        sentences = self._split_sentences(text)
        
        chunks = []
        current_chunk = ""
        chunk_start = 0
        
        for i, sentence in enumerate(sentences):
            # Try to add sentence to current chunk
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                # Current chunk is full, save it
                if current_chunk.strip():
                    chunk_data = self._create_chunk_data(
                        current_chunk.strip(), 
                        chunk_start, 
                        i - 1, 
                        metadata
                    )
                    chunks.append(chunk_data)
                
                # Start new chunk with overlap
                current_chunk = sentence + " "
                chunk_start = i
        
        # Add the last chunk
        if current_chunk.strip():
            chunk_data = self._create_chunk_data(
                current_chunk.strip(), 
                chunk_start, 
                len(sentences) - 1, 
                metadata
            )
            chunks.append(chunk_data)
        
        # Post-process chunks to ensure proper overlap
        chunks = self._apply_overlap(chunks)
        
        logger.debug(f"Created {len(chunks)} chunks from text of length {len(text)}")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with chunking
        text = re.sub(r'[\r\n\t]', ' ', text)
        
        # Normalize quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('–', '-').replace('—', '-')
        
        return text.strip()
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - can be enhanced with more sophisticated NLP
        sentence_endings = r'[.!?]+'
        sentences = re.split(sentence_endings, text)
        
        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter out very short fragments
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _create_chunk_data(self, text: str, start_idx: int, end_idx: int, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create chunk data structure."""
        chunk_data = {
            'text': text,
            'start_sentence': start_idx,
            'end_sentence': end_idx,
            'length': len(text),
            'chunk_id': f"chunk_{start_idx}_{end_idx}"
        }
        
        # Add metadata if provided
        if metadata:
            chunk_data.update(metadata)
        
        return chunk_data
    
    def _apply_overlap(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply overlap between chunks."""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk - no overlap needed
                overlapped_chunks.append(chunk)
                continue
            
            # Add overlap from previous chunk
            prev_chunk = chunks[i - 1]
            overlap_text = self._get_overlap_text(prev_chunk['text'], chunk['text'])
            
            if overlap_text:
                # Create new chunk with overlap
                overlapped_chunk = chunk.copy()
                overlapped_chunk['text'] = overlap_text + " " + chunk['text']
                overlapped_chunk['length'] = len(overlapped_chunk['text'])
                overlapped_chunk['overlap_text'] = overlap_text
                overlapped_chunks.append(overlapped_chunk)
            else:
                overlapped_chunks.append(chunk)
        
        return overlapped_chunks
    
    def _get_overlap_text(self, prev_text: str, current_text: str) -> str:
        """Get overlap text between two chunks."""
        if not prev_text or not current_text:
            return ""
        
        # Find common words at the end of previous chunk and beginning of current
        prev_words = prev_text.split()[-self.overlap//10:]  # Last few words
        current_words = current_text.split()[:self.overlap//10]  # First few words
        
        # Find common sequence
        overlap_words = []
        for i in range(min(len(prev_words), len(current_words))):
            if prev_words[-(i+1)] == current_words[i]:
                overlap_words.insert(0, prev_words[-(i+1)])
            else:
                break
        
        return " ".join(overlap_words)
    
    def chunk_email_content(self, email_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk email content with metadata."""
        # Combine text and HTML content
        content = email_data.get('content', '')
        html_content = email_data.get('html_content', '')
        
        # Prefer HTML content if available, fall back to plain text
        if html_content and len(html_content) > len(content):
            text_to_chunk = html_content
        else:
            text_to_chunk = content
        
        # Create metadata for chunks
        metadata = {
            'file_path': email_data.get('file_path', ''),
            'subject': email_data.get('subject', ''),
            'sender': email_data.get('sender', ''),
            'date': email_data.get('date_received', ''),
            'msg_id': email_data.get('msg_id', ''),
            'email_type': 'html' if html_content and len(html_content) > len(content) else 'text'
        }
        
        return self.chunk_text(text_to_chunk, metadata)
