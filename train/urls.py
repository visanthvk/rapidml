from django.conf.urls import url
from . import views

urlpatterns = [
    url('^$', views.index),
    url('trainmodel', views.train),
]