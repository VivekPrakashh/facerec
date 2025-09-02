from rest_framework import serializers
from .models import KnownFace

class KnownFaceSerializer(serializers.ModelSerializer):
    class Meta:
        model = KnownFace
        fields = ['id', 'name', 'image']
