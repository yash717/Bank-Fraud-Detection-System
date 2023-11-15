
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('fraud/', include('bank_apis.urls')),  # Include ml_app's URL patterns
]
