
import insightface
from sklearn.preprocessing import Normalizer
from mtcnn import MTCNN

print("📂 Initializing InsightFace (ArcFace + RetinaFace) via insightface-rk...")

from insightface.app import FaceAnalysis
model = FaceAnalysis(name="buffalo_l")  
model.prepare(ctx_id=0)

print("✅ insightface-rk model ready.")\

detector = MTCNN()

l2_normalizer = Normalizer(norm='l2')


def get_facenet_model():
    """
    Returns the fully prepared FaceAnalysis model (detection + embedding).
    """
    return model

def get_face_detector():
    return detector


def get_normalizer():
    """
    Returns the L2 normalizer for embedding vectors.
    """
    return l2_normalizer

