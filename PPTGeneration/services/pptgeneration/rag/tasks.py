import time
import logging
import json
import uuid
from django.conf import settings
from .parsing import EmailParser
from .chunking import TextChunker
from .qdrant_client_ext import QdrantClientExt
from .ollama_client import OllamaClient
from .prompting import PromptManager
from .ppt_builder import PPTBuilder

logger = logging.getLogger(__name__)


def reindex_emails(force=False):
    """Reindex all email files in the data directory."""
    try:
        start_time = time.time()
        
        # Initialize components
        parser = EmailParser()
        chunker = TextChunker()
        qdrant_client = QdrantClientExt()
        ollama_client = OllamaClient()
        
        # Parse and process emails
        emails = parser.parse_all_emails(force=force)
        
        total_chunks = 0
        for email_data in emails:
            # 1) Text chunks → text embeddings
            chunks = chunker.chunk_text(email_data['content_text'])
            for ch in chunks:
                ch['embedding_text'] = ollama_client.embeddings(
                    model="nomic-embed-text",
                    prompt=ch['text']
                )["embedding"]
                ch["payload"] = {
                    "msg_id": email_data["msg_id"],
                    "date_received_ts": email_data["date_received"],
                    "chunk_id": ch.get("id") or str(uuid.uuid4())
                }

            # 2) Images (base64 list) → caption/OCR with Llama 3.2 Vision
            image_items = []
            for img_item in email_data.get('content_images', []):
                try:
                    # Use the new analyze_image method
                    analysis_result = ollama_client.analyze_image(
                        image_base64=img_item['base64_data']
                    )
                    
                    # Generate embedding for caption
                    cap_emb = ollama_client.embeddings(
                        model="nomic-embed-text",
                        prompt=analysis_result["caption"]
                    )["embedding"]
                    
                    # Create image item with all metadata
                    item = {
                        "mime_type": img_item.get('mime_type', ''),
                        "base64_data": img_item.get('base64_data', ''),
                        "size_bytes": img_item.get('size_bytes', 0),
                        "is_embedded": img_item.get('is_embedded', True),
                        "caption": analysis_result["caption"],
                        "ocr": analysis_result.get("ocr", ""),
                        "entities": analysis_result.get("entities", []),
                        "embedding_text": cap_emb
                    }
                    image_items.append(item)
                    
                except Exception as e:
                    logger.error(f"Failed to process image: {e}")
                    # Add fallback item
                    item = {
                        "mime_type": img_item.get('mime_type', ''),
                        "base64_data": img_item.get('base64_data', ''),
                        "size_bytes": img_item.get('size_bytes', 0),
                        "is_embedded": img_item.get('is_embedded', True),
                        "caption": "Image processing failed",
                        "ocr": "",
                        "entities": [],
                        "embedding_text": []
                    }
                    image_items.append(item)

            email_data["chunks"] = chunks
            email_data["images"] = image_items

            # 3) Upsert into Qdrant
            qdrant_client.upsert_email_chunks(email_data)       # text collection
            qdrant_client.upsert_email_images(email_data)       # image collection

        processing_time = time.time() - start_time
        
        logger.info(f"Reindexing completed, {str({
            'emails_processed': len(emails),
            'total_chunks': total_chunks,
            'processing_time': processing_time,
            'force': force
        })}")
        
        return {
            'status': 'success',
            'emails_processed': len(emails),
            'total_chunks': total_chunks,
            'processing_time': processing_time
        }
        
    except Exception as e:
        logger.error(f"Reindexing failed: {e}")
        raise


def ask_question(question, top_k=6, include_sources=True):
    """Ask a question and get an answer using RAG."""
    try:
        start_time = time.time()
        
        # Initialize components
        qdrant_client = QdrantClientExt()
        ollama_client = OllamaClient()
        prompt_manager = PromptManager()
        
        # Generate question embedding
        question_embedding = ollama_client.generate_embedding(question)
        
        # Retrieve relevant chunks and images
        retrieval_start = time.time()
        relevant_chunks = qdrant_client.search_similar(question_embedding, top_k=top_k)
        relevant_images = qdrant_client.search_similar_images(question_embedding, top_k=3)  # Get top 3 relevant images
        retrieval_time = time.time() - retrieval_start
        
        if not relevant_chunks:
            return {
                'answer': 'Não encontrei informações relevantes nos emails para responder à sua pergunta. Sugiro verificar se os emails foram indexados corretamente.',
                'sources': [],
                'retrieval_time': retrieval_time,
                'generation_time': 0,
                'confidence': 0.0
            }
        
        # Build context from chunks and images
        context = prompt_manager.build_context(relevant_chunks, relevant_images)
        
        # Generate answer using LLaMA
        generation_start = time.time()
        answer = ollama_client.generate_answer(question, context)
        generation_time = time.time() - generation_start
        
        # Format sources if requested
        sources = []
        if include_sources:
            # Add text chunk sources
            for chunk in relevant_chunks:
                sources.append({
                    'type': 'text',
                    'subject': chunk.get('subject', 'Unknown'),
                    'date': chunk.get('date', 'Unknown'),
                    'file': chunk.get('file_path', 'Unknown'),
                    'content': chunk.get('text', '')[:200] + '...'
                })
            
            # Add image sources
            for image in relevant_images:
                sources.append({
                    'type': 'image',
                    'subject': image.get('subject', 'Unknown'),
                    'date': image.get('date', 'Unknown'),
                    'file': image.get('file_path', 'Unknown'),
                    'caption': image.get('caption', ''),
                    'ocr_text': image.get('ocr_text', '')[:200] + '...' if image.get('ocr_text') else ''
                })
        
        total_time = time.time() - start_time
        
        return {
            'answer': answer,
            'sources': sources,
            'retrieval_time': retrieval_time,
            'generation_time': generation_time,
            'confidence': 0.8  # Default confidence
        }
        
    except Exception as e:
        logger.error(f"Question answering failed: {e}")
        raise


def generate_ppt(question, slide_count=5, include_charts=True):
    """Generate PowerPoint presentation from email content."""
    try:
        start_time = time.time()
        
        # Get answer and context first
        rag_result = ask_question(question, top_k=10, include_sources=True)
        
        # Initialize PPT builder
        ppt_builder = PPTBuilder()
        
        # Generate PPT content
        ppt_content = ppt_builder.generate_content_from_rag(rag_result, slide_count, include_charts)
        
        # Build PPTX file
        pptx_data = ppt_builder.build_pptx(ppt_content)
        
        generation_time = time.time() - start_time
        
        logger.info(f"PPT generated successfully, {str({
            'slide_count': slide_count,
            'include_charts': include_charts,
            'generation_time': generation_time
        })}")
        
        return pptx_data
        
    except Exception as e:
        logger.error(f"PPT generation failed: {e}")
        raise
