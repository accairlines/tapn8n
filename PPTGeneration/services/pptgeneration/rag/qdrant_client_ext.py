import logging
import hashlib
from typing import List, Dict, Any, Optional
import httpx
from django.conf import settings

logger = logging.getLogger(__name__)


class QdrantClientExt:
    """Extended Qdrant client for email chunk storage and retrieval."""
    
    def __init__(self):
        self.base_url = settings.QDRANT_URL
        self.collection_name = "email_chunks"
        self.image_collection_name = "email_images"  # New collection for images
        self.vector_size = 768  # Default for nomic-embed-text
        self.http_client = httpx.Client(timeout=30.0)
        
        # Initialize collections
        self._ensure_collection_exists()
        self._ensure_image_collection_exists()
    
    def _ensure_collection_exists(self):
        """Ensure the email chunks collection exists."""
        try:
            # Check if collection exists
            response = self.http_client.get(f"{self.base_url}/collections/{self.collection_name}")
            
            if response.status_code == 404:
                # Create collection
                self._create_collection()
            elif response.status_code != 200:
                logger.error(f"Failed to check collection: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise
    
    def _ensure_image_collection_exists(self):
        """Ensure the email images collection exists."""
        try:
            # Check if collection exists
            response = self.http_client.get(f"{self.base_url}/collections/{self.image_collection_name}")
            
            if response.status_code == 404:
                # Create image collection
                self._create_image_collection()
            elif response.status_code != 200:
                logger.error(f"Failed to check image collection: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to ensure image collection exists: {e}")
            raise
    
    def _create_collection(self):
        """Create the email chunks collection."""
        try:
            collection_config = {
                "vectors": {
                    "size": self.vector_size,
                    "distance": "Cosine"
                },
                "on_disk_payload": True
            }
            
            response = self.http_client.put(
                f"{self.base_url}/collections/{self.collection_name}",
                json=collection_config
            )
            
            if response.status_code == 200:
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.error(f"Failed to create collection: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    def _create_image_collection(self):
        """Create the email images collection."""
        try:
            collection_config = {
                "vectors": {
                    "size": self.vector_size,
                    "distance": "Cosine"
                },
                "on_disk_payload": True
            }
            
            response = self.http_client.put(
                f"{self.base_url}/collections/{self.image_collection_name}",
                json=collection_config
            )
            
            if response.status_code == 200:
                logger.info(f"Created image collection: {self.image_collection_name}")
            else:
                logger.error(f"Failed to create image collection: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to create image collection: {e}")
            raise
    
    def upsert_email_chunks(self, email_data: Dict[str, Any]):
        """Upsert email chunks to Qdrant."""
        try:
            chunks = email_data.get('chunks', [])
            if not chunks:
                logger.warning(f"No chunks to upsert for email: {email_data.get('file_path', 'unknown')}")
                return
            
            # Prepare points for upsert
            points = []
            for chunk in chunks:
                point_id = self._generate_chunk_id(email_data, chunk)
                
                point = {
                    "id": point_id,
                    "vector": chunk.get('embedding', []),
                    "payload": {
                        "text": chunk.get('text', ''),
                        "file_path": email_data.get('file_path', ''),
                        "subject": email_data.get('subject', ''),
                        "sender": email_data.get('sender', ''),
                        "date": str(email_data.get('date_received', '')),
                        "msg_id": email_data.get('msg_id', ''),
                        "chunk_id": chunk.get('chunk_id', ''),
                        "chunk_length": chunk.get('length', 0),
                        "email_type": chunk.get('email_type', 'text')
                    }
                }
                points.append(point)
            
            # Upsert points
            response = self.http_client.put(
                f"{self.base_url}/collections/{self.collection_name}/points",
                json={"points": points}
            )
            
            if response.status_code == 200:
                logger.info(f"Upserted {len(points)} chunks for email: {email_data.get('file_path', 'unknown')}")
            else:
                logger.error(f"Failed to upsert chunks: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to upsert email chunks: {e}")
            raise
    
    def upsert_email_images(self, email_data: Dict[str, Any]):
        """Upsert email images to Qdrant."""
        try:
            images = email_data.get('images', [])
            if not images:
                logger.warning(f"No images to upsert for email: {email_data.get('file_path', 'unknown')}")
                return
            
            # Prepare points for upsert
            points = []
            for image in images:
                point_id = self._generate_image_id(email_data, image)
                
                point = {
                    "id": point_id,
                    "vector": image.get('embedding_text', []),
                    "payload": {
                        "file_path": email_data.get('file_path', ''),
                        "subject": email_data.get('subject', ''),
                        "sender": email_data.get('sender', ''),
                        "date": str(email_data.get('date_received', '')),
                        "msg_id": email_data.get('msg_id', ''),
                        "mime_type": image.get('mime_type', ''),
                        "caption": image.get('caption', ''),
                        "ocr_text": image.get('ocr', ''),
                        "entities": image.get('entities', []),
                        "size_bytes": image.get('size_bytes', 0),
                        "is_embedded": image.get('is_embedded', False)
                    }
                }
                points.append(point)
            
            # Upsert points
            response = self.http_client.put(
                f"{self.base_url}/collections/{self.image_collection_name}/points",
                json={"points": points}
            )
            
            if response.status_code == 200:
                logger.info(f"Upserted {len(points)} images for email: {email_data.get('file_path', 'unknown')}")
            else:
                logger.error(f"Failed to upsert images: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to upsert email images: {e}")
            raise
    
    def search_similar(self, query_vector: List[float], top_k: int = 6, score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity."""
        try:
            search_request = {
                "vector": query_vector,
                "limit": top_k,
                "score_threshold": score_threshold,
                "with_payload": True,
                "with_vector": False
            }
            
            response = self.http_client.post(
                f"{self.base_url}/collections/{self.collection_name}/points/search",
                json=search_request
            )
            
            if response.status_code == 200:
                results = response.json()
                chunks = []
                
                for result in results.get('result', []):
                    chunk_data = {
                        'text': result['payload'].get('text', ''),
                        'file_path': result['payload'].get('file_path', ''),
                        'subject': result['payload'].get('subject', ''),
                        'sender': result['payload'].get('sender', ''),
                        'date': result['payload'].get('date', ''),
                        'msg_id': result['payload'].get('msg_id', ''),
                        'chunk_id': result['payload'].get('chunk_id', ''),
                        'score': result.get('score', 0.0)
                    }
                    chunks.append(chunk_data)
                
                logger.debug(f"Found {len(chunks)} similar chunks")
                return chunks
            else:
                logger.error(f"Search failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_similar_images(self, query_vector: List[float], top_k: int = 6, score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar images using vector similarity."""
        try:
            search_request = {
                "vector": query_vector,
                "limit": top_k,
                "score_threshold": score_threshold,
                "with_payload": True,
                "with_vector": False
            }
            
            response = self.http_client.post(
                f"{self.base_url}/collections/{self.image_collection_name}/points/search",
                json=search_request
            )
            
            if response.status_code == 200:
                results = response.json()
                images = []
                
                for result in results.get('result', []):
                    image_data = {
                        'file_path': result['payload'].get('file_path', ''),
                        'subject': result['payload'].get('subject', ''),
                        'sender': result['payload'].get('sender', ''),
                        'date': result['payload'].get('date', ''),
                        'msg_id': result['payload'].get('msg_id', ''),
                        'mime_type': result['payload'].get('mime_type', ''),
                        'caption': result['payload'].get('caption', ''),
                        'ocr_text': result['payload'].get('ocr_text', ''),
                        'entities': result['payload'].get('entities', []),
                        'score': result.get('score', 0.0)
                    }
                    images.append(image_data)
                
                logger.debug(f"Found {len(images)} similar images")
                return images
            else:
                logger.error(f"Image search failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Image search failed: {e}")
            return []
    
    def delete_email_chunks(self, file_path: str):
        """Delete all chunks for a specific email file."""
        try:
            # Find chunks by file_path
            filter_request = {
                "filter": {
                    "must": [
                        {
                            "key": "file_path",
                            "match": {"value": file_path}
                        }
                    ]
                }
            }
            
            response = self.http_client.post(
                f"{self.base_url}/collections/{self.collection_name}/points/scroll",
                json=filter_request
            )
            
            if response.status_code == 200:
                results = response.json()
                point_ids = [point['id'] for point in results.get('result', [])]
                
                if point_ids:
                    # Delete points
                    delete_response = self.http_client.post(
                        f"{self.base_url}/collections/{self.collection_name}/points/delete",
                        json={"points": point_ids}
                    )
                    
                    if delete_response.status_code == 200:
                        logger.info(f"Deleted {len(point_ids)} chunks for file: {file_path}")
                    else:
                        logger.error(f"Failed to delete chunks: {delete_response.status_code}")
                else:
                    logger.info(f"No chunks found to delete for file: {file_path}")
            else:
                logger.error(f"Failed to find chunks for deletion: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to delete email chunks: {e}")
            raise
    
    def delete_email_images(self, file_path: str):
        """Delete all images for a specific email file."""
        try:
            # Find images by file_path
            filter_request = {
                "filter": {
                    "must": [
                        {
                            "key": "file_path",
                            "match": {"value": file_path}
                        }
                    ]
                }
            }
            
            response = self.http_client.post(
                f"{self.base_url}/collections/{self.image_collection_name}/points/scroll",
                json=filter_request
            )
            
            if response.status_code == 200:
                results = response.json()
                point_ids = [point['id'] for point in results.get('result', [])]
                
                if point_ids:
                    # Delete points
                    delete_response = self.http_client.post(
                        f"{self.base_url}/collections/{self.image_collection_name}/points/delete",
                        json={"points": point_ids}
                    )
                    
                    if delete_response.status_code == 200:
                        logger.info(f"Deleted {len(point_ids)} images for file: {file_path}")
                    else:
                        logger.error(f"Failed to delete images: {delete_response.status_code}")
                else:
                    logger.info(f"No images found to delete for file: {file_path}")
            else:
                logger.error(f"Failed to find images for deletion: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to delete email images: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information and statistics."""
        try:
            response = self.http_client.get(f"{self.base_url}/collections/{self.collection_name}")
            
            if response.status_code == 200:
                collection_info = response.json()
                
                # Get collection size
                size_response = self.http_client.get(f"{self.base_url}/collections/{self.collection_name}/points/count")
                if size_response.status_code == 200:
                    size_info = size_response.json()
                    collection_info['point_count'] = size_info.get('result', {}).get('count', 0)
                
                return collection_info
            else:
                logger.error(f"Failed to get collection info: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}
    
    def health_check(self) -> Dict[str, Any]:
        """Check Qdrant service health."""
        try:
            response = self.http_client.get(f"{self.base_url}/readyz")
            
            if response.status_code == 200:
                health_info = response.json()
                return {
                    'status': 'readyz',
                    'version': health_info.get('version', 'unknown'),
                    'uptime': health_info.get('uptime', 0)
                }
            else:
                return {
                    'status': 'unreadyz',
                    'error': f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {
                'status': 'unreadyz',
                'error': str(e)
            }
    
    def _generate_chunk_id(self, email_data: Dict[str, Any], chunk: Dict[str, Any]) -> str:
        """Generate a deterministic ID for a chunk."""
        # Create a hash from file path and chunk content
        content = f"{email_data.get('file_path', '')}_{chunk.get('chunk_id', '')}_{chunk.get('text', '')[:100]}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _generate_image_id(self, email_data: Dict[str, Any], image: Dict[str, Any]) -> str:
        """Generate a deterministic ID for an image."""
        # Create a hash from file path and image metadata
        content = f"{email_data.get('file_path', '')}_{image.get('mime_type', '')}_{image.get('size_bytes', 0)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def __del__(self):
        """Cleanup HTTP client."""
        if hasattr(self, 'http_client'):
            self.http_client.close()
