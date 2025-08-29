import time
import base64
import logging
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView
from .permissions import HasValidAPIKey

from .serializers import (
    ReindexRequestSerializer, QuestionRequestSerializer, 
    PPTGenerationRequestSerializer
)
from .tasks import reindex_emails, ask_question, generate_ppt
from .office365_sync import Office365EmailSync

logger = logging.getLogger(__name__)


@api_view(['GET'])
@permission_classes([AllowAny])
def health_check(request):
    """Health check endpoint to verify service status."""
    try:
        # Check Qdrant connection
        from .qdrant_client_ext import QdrantClientExt
        qdrant_client = QdrantClientExt()
        qdrant_status = qdrant_client.health_check()
        
        # Check LLaMA connection
        from .ollama_client import OllamaClient
        ollama_client = OllamaClient()
        llama_status = ollama_client.health_check()
        
        return Response({
            'status': 'healthy',
            'qdrant': qdrant_status,
            'llama': llama_status,
            'timestamp': time.time()
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return Response({
            'status': 'unhealthy',
            'error': str(e)
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)

@api_view(['POST'])
@permission_classes([HasValidAPIKey])
def reindex(request):
    serializer = ReindexRequestSerializer(data=request.data)
    if serializer.is_valid():
        force = serializer.validated_data.get('force', False)
        
        try:
            # Start async reindexing task
            task_id = reindex_emails.delay(force=force)
            
            logger.info(f"Reindexing triggered {'task_id': str(task_id), 'force': force}")
            
            return Response({
                'message': 'Reindexing started',
                'task_id': str(task_id),
                'force': force
            }, status=status.HTTP_202_ACCEPTED)
        except Exception as e:
            logger.error(f"Failed to start reindexing: {e}")
            return Response({
                'error': 'Failed to start reindexing',
                'details': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@permission_classes([HasValidAPIKey])
def ask(request):
    serializer = QuestionRequestSerializer(data=request.data)
    if serializer.is_valid():
        question = serializer.validated_data['question']
        top_k = serializer.validated_data.get('top_k', settings.TOP_K_DEFAULT)
        include_sources = serializer.validated_data.get('include_sources', True)
        
        try:
            start_time = time.time()
            
            # Get answer using RAG
            result = ask_question(question, top_k=top_k, include_sources=include_sources)
            
            end_time = time.time()
            latency = {
                'total_time': end_time - start_time,
                'retrieval_time': result.get('retrieval_time', 0),
                'generation_time': result.get('generation_time', 0)
            }
            
            response_data = {
                'answer': result['answer'],
                'sources': result.get('sources', []),
                'latency': latency,
                'confidence': result.get('confidence', 0.8)
            }
            
            logger.info(f"Question answered: {question[:100]}..., {str({
                'question_length': len(question),
                'top_k': top_k,
                'latency': latency
            })}")
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            return Response({
                'error': 'Failed to answer question',
                'details': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@permission_classes([HasValidAPIKey])
def generate_ppt(request):
    serializer = PPTGenerationRequestSerializer(data=request.data)
    if serializer.is_valid():
        question = serializer.validated_data['question']
        slide_count = serializer.validated_data.get('slide_count', 5)
        include_charts = serializer.validated_data.get('include_charts', True)
        
        try:
            start_time = time.time()
            
            # Generate PPT
            pptx_data = generate_ppt(question, slide_count=slide_count, include_charts=include_charts)
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            # Encode PPTX as base64
            pptx_base64 = base64.b64encode(pptx_data).decode('utf-8')
            
            response_data = {
                'pptx_base64': pptx_base64,
                'filename': f"email_analysis_{int(time.time())}.pptx",
                'slide_count': slide_count,
                'generation_time': generation_time
            }
            
            logger.info(f"PPT generated for question: {question[:100]}..., {str({
                'slide_count': slide_count,
                'include_charts': include_charts,
                'generation_time': generation_time
            })}")
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Failed to generate PPT: {e}")
            return Response({
                'error': 'Failed to generate PPT',
                'details': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@permission_classes([HasValidAPIKey])
def sync_emails(request):
    """Trigger email synchronization with POST data."""
    try:
        # Access the JSON body data from POST request
        days = request.data.get('days')
        if days is None:
            # Initialize email sync
            email_sync = Office365EmailSync()
            # Get sync status if days not provided
            status_info = email_sync.get_sync_status()
            return Response(status_info, status=status.HTTP_200_OK)
        
        # Validate days parameter
        if not isinstance(days, int) or days < 1 or days > 830:
            return Response({
                'error': 'Invalid days parameter. Must be between 1 and 365.'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Initialize email sync
        email_sync = Office365EmailSync()
        
        # Perform sync
        start_time = time.time()
        sync_results = email_sync.sync_emails(days=days)
        end_time = time.time()
        
        sync_time = end_time - start_time
        
        logger.info(f"Email sync completed, {str({
            'days': days,
            'sync_results': sync_results,
            'sync_time': sync_time
        })}")
        
        return Response({
            'message': 'Email synchronization completed',
            'days_processed': days,
            'results': sync_results,
            'sync_time_seconds': sync_time,
            'timestamp': time.time()
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Email sync failed: {e}")
        return Response({
            'error': 'Email synchronization failed',
            'details': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)