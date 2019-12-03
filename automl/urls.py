
from django.contrib import admin
from django.urls import path
from django.conf.urls import url, include
from automl import settings
from django.conf.urls.static import static

urlpatterns = [
    url('home/', include('home.urls')),
    url('train/', include('train.urls')),
    url('deploy/', include('deploy.urls')),
    url('predict/', include('predict.urls')),
    url('preprocess/', include('preprocess.urls')),
    url('tune/', include('tune.urls')),
    url('visualise/', include('visualise.urls')),
    url('^$', include('home.urls')),
    path('admin/', admin.site.urls),
]+static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)