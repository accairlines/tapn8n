import logging
import json
from typing import List, Dict, Any, Optional
import httpx
from django.conf import settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama LLaMA service."""
    
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.LLAMA_MODEL
        self.embed_model = settings.EMBED_MODEL
        self.http_client = httpx.Client(timeout=120.0)  # Longer timeout for generation
        
        # Test connection on initialization
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Ollama service."""
        try:
            response = self.http_client.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                logger.warning(f"Ollama service may not be available: {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not connect to Ollama service: {e}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate text embedding using the specified model."""
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for embedding")
                return []
            
            request_data = {
                "model": self.embed_model,
                "prompt": text.strip()
            }
            
            response = self.http_client.post(
                f"{self.base_url}/api/embeddings",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = result.get('embedding', [])
                
                if not embedding:
                    logger.error("No embedding returned from Ollama")
                    return []
                
                logger.debug(f"Generated embedding of size {len(embedding)} for text: {text[:100]}...")
                return embedding
            else:
                logger.error(f"Embedding generation failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []
    
    def generate_answer(self, question: str, context: str, max_tokens: int = 1000) -> str:
        """Generate answer using LLaMA model."""
        try:
            # Build the complete prompt
            prompt = self._build_question_prompt(question, context)
            
            request_data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            }
            
            response = self.http_client.post(
                f"{self.base_url}/api/chat",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                
                if not answer:
                    logger.warning("Empty response from LLaMA")
                    return "Não consegui gerar uma resposta adequada. Por favor, tente reformular a pergunta."
                
                logger.debug(f"Generated answer of length {len(answer)}")
                return answer
            else:
                logger.error(f"Answer generation failed: {response.status_code}")
                return "Erro ao gerar resposta. Por favor, tente novamente."
                
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"Erro interno: {str(e)}"
    
    def generate_ppt_content(self, question: str, context: str, slide_count: int = 5) -> str:
        """Generate structured content for PowerPoint presentation."""
        try:
            prompt = self._build_ppt_prompt(question, context, slide_count)
            
            request_data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 2000,  # Longer response for PPT content
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            }
            
            response = self.http_client.post(
                f"{self.base_url}/api/chat",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('response', '').strip()
                
                if not content:
                    logger.warning("Empty PPT content from LLaMA")
                    return "Não consegui gerar conteúdo para a apresentação."
                
                logger.debug(f"Generated PPT content of length {len(content)}")
                return content
            else:
                logger.error(f"PPT content generation failed: {response.status_code}")
                return "Erro ao gerar conteúdo da apresentação."
                
        except Exception as e:
            logger.error(f"PPT content generation failed: {e}")
            return f"Erro interno: {str(e)}"
    
    def _build_question_prompt(self, question: str, context: str) -> str:
        """Build prompt for question answering."""
        prompt = f"""Instruções: Responde apenas com base nos emails indexados; cita as fontes no formato [assunto | data | ficheiro]; se não houver evidência suficiente, diz que não sabes e sugere próximos passos; sê conciso.

Pergunta: {question}

Contexto dos emails:
{context}

Resposta:"""
        return prompt
    
    def _build_ppt_prompt(self, question: str, context: str, slide_count: int) -> str:
        """Build prompt for PPT content generation."""
        prompt = f"""Com base na informação dos emails fornecida, cria um resumo estruturado para uma apresentação PowerPoint.

Instruções:
1. Organiza a informação de forma lógica e clara
2. Identifica os pontos principais e secundários
3. Sugere títulos de slides apropriados
4. Inclui dados quantitativos se disponíveis
5. Mantém um tom profissional e objetivo
6. Cita as fontes dos emails quando relevante

Pergunta/Contexto: {question}
Número de slides: {slide_count}

Informação dos emails:
{context}

Formato de resposta:
- Título da apresentação
- Lista de slides com títulos e conteúdo resumido
- Pontos-chave para cada slide
- Recomendações ou conclusões

Resposta estruturada:"""
        return prompt
    
    def health_check(self) -> Dict[str, Any]:
        """Check Ollama service health."""
        try:
            response = self.http_client.get(f"{self.base_url}/api/tags")
            
            if response.status_code == 200:
                models_info = response.json()
                available_models = [model['name'] for model in models_info.get('models', [])]
                
                return {
                    'status': 'healthy',
                    'available_models': available_models,
                    'target_model': self.model,
                    'embed_model': self.embed_model
                }
            else:
                return {
                    'status': 'unhealthy',
                    'error': f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def get_model_info(self, model_name: str = None) -> Dict[str, Any]:
        """Get information about a specific model."""
        try:
            model = model_name or self.model
            response = self.http_client.post(
                f"{self.base_url}/api/show",
                json={"name": model}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get model info: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {}
    
    def __del__(self):
        """Cleanup HTTP client."""
        if hasattr(self, 'http_client'):
            self.http_client.close()

    def analyze_image(self, image_base64: str, prompt: str = None) -> Dict[str, Any]:
        """Analyze an image using vision model."""
        try:
            if not prompt:
                prompt = "Describe the image, extract any visible text, and list key entities. Return strict JSON with keys: caption, ocr, entities[]."
            
            messages = [{
                "role": "user",
                "content": prompt,
                "images": [image_base64]
            }]
            
            response = self.chat(
                model="llama3.2-vision",  # Use vision model
                messages=messages,
                stream=False
            )
            
            if response and 'message' in response:
                content = response['message'].get('content', '')
                
                # Try to parse JSON response
                try:
                    import json
                    data = json.loads(content)
                    return {
                        'success': True,
                        'caption': data.get('caption', ''),
                        'ocr': data.get('ocr', ''),
                        'entities': data.get('entities', []),
                        'raw_response': content
                    }
                except json.JSONDecodeError:
                    # Fallback to raw response
                    return {
                        'success': False,
                        'caption': content,
                        'ocr': '',
                        'entities': [],
                        'raw_response': content,
                        'error': 'Failed to parse JSON response'
                    }
            else:
                return {
                    'success': False,
                    'caption': 'Failed to analyze image',
                    'ocr': '',
                    'entities': [],
                    'raw_response': '',
                    'error': 'No response from vision model'
                }
                
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {
                'success': False,
                'caption': f'Error: {str(e)}',
                'ocr': '',
                'entities': [],
                'raw_response': '',
                'error': str(e)
            }

    def embeddings(self, model: str = None, prompt: str = "") -> dict:
        """
        Compatibility wrapper so callers can use `.embeddings(model=..., prompt=...)`.
        Returns a dict with key 'embedding' like Ollama HTTP API.
        """
        # Optionally honor the model arg; we already store embed_model on self
        if model and model != self.embed_model:
            # You can log a warning or just ignore; here we ignore and use self.embed_model
            pass
        emb = self.generate_embedding(prompt)
        return {"embedding": emb}

    def chat(self, model: str = None, messages: list = None, options: dict = None, stream: bool = False) -> dict:
        """
        Thin wrapper over /api/chat to support vision (base64) prompts.
        Expects messages=[{role, content, images?}], returns the API JSON.
        """
        payload = {
            "model": model or self.model,
            "messages": messages or [],
            "stream": stream
        }
        if options:
            payload["options"] = options

        resp = self.http_client.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()
