from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('canvas/', views.canvas, name='canvas'),
    path('about/', views.about, name='about'),
    path('canvas/output',views.output,name='output'),
]
