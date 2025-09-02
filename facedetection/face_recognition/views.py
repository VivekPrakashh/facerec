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

def index(request):
    return render(request, 'index.html')

# Load models once
facenet_model = get_facenet_model()
detector = get_face_detector()
l2_normalizer = get_normalizer()

def extract_face(image_path, required_size=(160, 160)):
    image = Image.open(image_path).convert('RGB')
    pixels = np.asarray(image)
    results = detector.detect_faces(pixels)
    if not results:
        print("❌ No face detected.")
        return None, None
    x1, y1, width, height = results[0]['box']
    confidence = results[0].get('confidence', None)
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face).resize(required_size)
    face_array = np.asarray(image)
    print("✅ Face shape:", face_array.shape)
    return face_array, confidence

def get_embedding(face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    embedding = facenet_model.embeddings(samples)[0]
    print("Embedding shape:", embedding.shape)
    print("✅ Embedding sample:", embedding[:5])
    return embedding

# ENROLL FACE VIEW
class EnrollFaceView(APIView):
    def post(self, request):
        name = request.data.get('name')
        image_file = request.FILES.get('image')

        if not name or not image_file:
            return Response({"error": "Name and image are required"}, status=status.HTTP_400_BAD_REQUEST)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            for chunk in image_file.chunks():
                temp_file.write(chunk)
            image_path = temp_file.name

        face, confidence = extract_face(image_path)
        os.remove(image_path)

        if face is None:
            return Response({"error": "No face detected"}, status=status.HTTP_400_BAD_REQUEST)

        embedding = get_embedding(face)
        embedding = l2_normalizer.transform([embedding])[0].astype(np.float32)

        KnownFace.objects.create(
            name=name,
            embedding=embedding.tobytes()
        )

        return Response({
            "message": f"✅ Face enrolled for '{name}'",
            "detection_confidence": float(confidence)
        }, status=status.HTTP_201_CREATED)

# MATCH FACE VIEW
class MatchFaceView(APIView):
    def post(self, request):
        image_file = request.FILES.get('image')
        if not image_file:
            return Response({"error": "Image is required"}, status=status.HTTP_400_BAD_REQUEST)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            for chunk in image_file.chunks():
                temp_file.write(chunk)
            image_path = temp_file.name

        face, confidence = extract_face(image_path)
        os.remove(image_path)

        if face is None:
            return Response({"error": "No face detected"}, status=status.HTTP_400_BAD_REQUEST)

        embedding = get_embedding(face)
        embedding = l2_normalizer.transform([embedding])[0].astype(np.float32)

        print("🔍 Uploaded embedding sample:", embedding[:5])

        min_dist = float("inf")
        identity = "unknown"

        for known in KnownFace.objects.all():
            db_emb = np.frombuffer(known.embedding, dtype=np.float32)
            print(f"🔍 Comparing with {known.name}, DB embedding sample:", db_emb[:5])
            dist = np.linalg.norm(embedding - db_emb)
            print(f"Distance to {known.name}: {dist}")
            if dist < min_dist:
                min_dist = dist
                identity = known.name

        if min_dist > 0.8:
            identity = "unknown"

        return Response({
            "identity": identity,
            "distance": float(min_dist),
            "detection_confidence": float(confidence)
        })
