# face_recognition/utils.py
import os
from mtcnn import MTCNN
from sklearn.preprocessing import Normalizer
from keras_facenet import FaceNet

# Initialize FaceNet embedder
print("📂 Initializing FaceNet embedder...")
embedder = FaceNet()
print("✅ FaceNet embedder ready.")

# Face detector and normalizer
detector = MTCNN()
l2_normalizer = Normalizer(norm='l2')

def get_facenet_model():
    return embedder

def get_face_detector():
    return detector

def get_normalizer():
    return l2_normalizer
