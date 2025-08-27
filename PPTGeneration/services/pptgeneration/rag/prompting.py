import logging
from typing import List, Dict, Any
from django.conf import settings

logger = logging.getLogger(__name__)


class PromptManager:
    """Manages prompts and context building for RAG operations."""
    
    def __init__(self):
        self.system_prompt = """Responde apenas com base nos emails indexados; cita as fontes no formato [assunto | data | ficheiro]; se não houver evidência suficiente, diz que não sabes e sugere próximos passos; sê conciso."""
        
        self.ppt_generation_prompt = """Com base na informação dos emails fornecida, cria um resumo estruturado para uma apresentação PowerPoint. 
        
        Instruções:
        1. Organiza a informação de forma lógica e clara
        2. Identifica os pontos principais e secundários
        3. Sugere títulos de slides apropriados
        4. Inclui dados quantitativos se disponíveis
        5. Mantém um tom profissional e objetivo
        6. Cita as fontes dos emails quando relevante
        
        Formato de resposta:
        - Título da apresentação
        - Lista de slides com títulos e conteúdo resumido
        - Pontos-chave para cada slide
        - Recomendações ou conclusões"""
    
    def build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved chunks."""
        if not chunks:
            return ""
        
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            # Format chunk with metadata
            chunk_text = chunk.get('text', '')
            subject = chunk.get('subject', 'Assunto desconhecido')
            date = chunk.get('date', 'Data desconhecida')
            file_path = chunk.get('file_path', 'Ficheiro desconhecido')
            
            # Format date if it's a datetime object
            if hasattr(date, 'strftime'):
                date_str = date.strftime('%d/%m/%Y %H:%M')
            else:
                date_str = str(date)
            
            # Create formatted chunk
            formatted_chunk = f"[CHUNK {i+1}]\n"
            formatted_chunk += f"Fonte: {subject} | {date_str} | {file_path}\n"
            formatted_chunk += f"Conteúdo: {chunk_text}\n"
            formatted_chunk += "-" * 80 + "\n"
            
            context_parts.append(formatted_chunk)
        
        return "\n".join(context_parts)
    
    def build_question_prompt(self, question: str, context: str) -> str:
        """Build the complete prompt for question answering."""
        prompt = f"{self.system_prompt}\n\n"
        prompt += f"Pergunta: {question}\n\n"
        prompt += "Contexto dos emails:\n"
        prompt += context + "\n\n"
        prompt += "Resposta:"
        
        return prompt
    
    def build_ppt_generation_prompt(self, question: str, context: str, slide_count: int = 5) -> str:
        """Build the complete prompt for PPT generation."""
        prompt = f"{self.ppt_generation_prompt}\n\n"
        prompt += f"Pergunta/Contexto: {question}\n"
        prompt += f"Número de slides: {slide_count}\n\n"
        prompt += "Informação dos emails:\n"
        prompt += context + "\n\n"
        prompt += "Resposta estruturada:"
        
        return prompt
    
    def format_sources(self, chunks: List[Dict[str, Any]]) -> str:
        """Format source information for citations."""
        if not chunks:
            return "Sem fontes disponíveis."
        
        sources = []
        for chunk in chunks:
            subject = chunk.get('subject', 'Assunto desconhecido')
            date = chunk.get('date', 'Data desconhecida')
            file_path = chunk.get('file_path', 'Ficheiro desconhecido')
            
            # Format date
            if hasattr(date, 'strftime'):
                date_str = date.strftime('%d/%m/%Y')
            else:
                date_str = str(date)
            
            source = f"[{subject} | {date_str} | {file_path}]"
            sources.append(source)
        
        return " ".join(sources)
    
    def get_summary_prompt(self, content: str, max_length: int = 500) -> str:
        """Get prompt for content summarization."""
        prompt = f"Resume o seguinte conteúdo em no máximo {max_length} caracteres, mantendo os pontos principais:\n\n"
        prompt += content + "\n\n"
        prompt += "Resumo:"
        
        return prompt
    
    def get_key_points_prompt(self, content: str, point_count: int = 5) -> str:
        """Get prompt for extracting key points."""
        prompt = f"Extrai os {point_count} pontos principais do seguinte conteúdo:\n\n"
        prompt += content + "\n\n"
        prompt += f"Pontos principais ({point_count}):\n"
        
        return prompt
