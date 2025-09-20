from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import KnownFace
from .utils import get_facenet_model, get_face_detector, get_normalizer
import numpy as np
from PIL import Image
import tempfile
import os
import cv2

def index(request):
    return render(request, 'index.html')

facenet_model = get_facenet_model()
detector = get_face_detector()
l2_normalizer = get_normalizer()


def get_embedding(image_path):
    """
    Extracts a normalized face embedding using InsightFace.
    image_path: path to image file (str).
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image file")

    faces = facenet_model.get(image)
    if not faces:
        raise ValueError("No face detected in given image")

    face_obj = faces[0]
    embedding = face_obj.embedding
    embedding = l2_normalizer.transform([embedding])[0]

    return embedding, face_obj.det_score


class EnrollFaceView(APIView):
    def post(self, request):
        name = request.data.get("name")
        image_file = request.FILES.get("image")

        if not name or not image_file:
            return Response(
                {"error": "Name and image are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            for chunk in image_file.chunks():
                temp_file.write(chunk)
            image_path = temp_file.name

        try:
            embedding, confidence = get_embedding(image_path)
        except Exception as e:
            os.remove(image_path)
            return Response(
                {"error": str(e)}, status=status.HTTP_400_BAD_REQUEST
            )
        
        print(f"Enroll Face Embedding -----------------------> {embedding}")

        os.remove(image_path)

        KnownFace.objects.create(
            name=name,
            embedding=embedding.astype(np.float32).tobytes(),
        )

        return Response(
            {
                "message": f"✅ Face enrolled for '{name}'",
                "detection_confidence": float(confidence),
            },
            status=status.HTTP_201_CREATED,
        )
    
    # For Matching Face with Database Images

class MatchFaceView(APIView):
    def post(self, request):
        image_file = request.FILES.get("image")
        if not image_file:
            return Response(
                {"error": "Image is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            for chunk in image_file.chunks():
                temp_file.write(chunk)
            image_path = temp_file.name

        try:
            embedding, detection_confidence = get_embedding(image_path)
        except Exception as e:
            os.remove(image_path)
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        os.remove(image_path)

        min_dist = float("inf")
        identity = "unknown"

        for known in KnownFace.objects.all():
            db_emb = np.frombuffer(known.embedding, dtype=np.float32)
            dist = np.linalg.norm(embedding - db_emb)
            if dist < min_dist:
                min_dist = dist
                identity = known.name

        threshold = 0.8
        if min_dist > threshold:
            return Response({"identity": "unknown"}, status=status.HTTP_200_OK)
        
        detection_confidence = 1 - (min_dist / threshold)

        return Response(
            {
                "identity": identity,
                "distance": float(min_dist),
                "detection_confidence": float(detection_confidence),
            },
            status=status.HTTP_200_OK,
        )

