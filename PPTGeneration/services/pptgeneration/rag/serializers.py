from rest_framework import serializers
from .models import EmailDocument, ProcessingLog


class EmailDocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = EmailDocument
        fields = '__all__'
        read_only_fields = ['processed_at', 'last_indexed']


class ProcessingLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProcessingLog
        fields = '__all__'
        read_only_fields = ['timestamp']


class ReindexRequestSerializer(serializers.Serializer):
    force = serializers.BooleanField(default=False, help_text="Force reindexing of all documents")


class QuestionRequestSerializer(serializers.Serializer):
    question = serializers.CharField(max_length=1000, help_text="Question to ask about the emails")
    top_k = serializers.IntegerField(default=6, min_value=1, max_value=20, help_text="Number of relevant chunks to retrieve")
    include_sources = serializers.BooleanField(default=True, help_text="Include source documents in response")


class QuestionResponseSerializer(serializers.Serializer):
    answer = serializers.CharField(help_text="Answer to the question")
    sources = serializers.ListField(child=serializers.DictField(), help_text="Source documents used")
    latency = serializers.DictField(help_text="Processing latency information")
    confidence = serializers.FloatField(min_value=0.0, max_value=1.0, help_text="Confidence score")


class PPTGenerationRequestSerializer(serializers.Serializer):
    question = serializers.CharField(max_length=1000, help_text="Question context for PPT generation")
    slide_count = serializers.IntegerField(default=5, min_value=1, max_value=20, help_text="Number of slides to generate")
    include_charts = serializers.BooleanField(default=True, help_text="Include charts and visualizations")


class PPTGenerationResponseSerializer(serializers.Serializer):
    pptx_base64 = serializers.CharField(help_text="Base64 encoded PPTX file")
    filename = serializers.CharField(help_text="Suggested filename for the PPTX")
    slide_count = serializers.IntegerField(help_text="Actual number of slides generated")
    generation_time = serializers.FloatField(help_text="Time taken to generate PPT in seconds")
