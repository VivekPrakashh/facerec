from django.urls import path
from .views import index, EnrollFaceView, MatchFaceView

urlpatterns = [
    path('', index, name='index'),
    path('api/enroll/', EnrollFaceView.as_view(), name='enroll'),
    path('api/recognize/', MatchFaceView.as_view(), name='recognize'),
]
