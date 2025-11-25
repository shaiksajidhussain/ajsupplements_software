from django.contrib import admin
from django.urls import path, include
from django.http import HttpResponse


# Simple test view (home)
def home(request):
    return HttpResponse("<h2>IMT Feed Formulation App</h2><p>Welcome! Go to /admin/ to manage data.</p>")

urlpatterns = [
    path("admin/", admin.site.urls),


    # All app routes go here
    path("", include("core.urls")),  # ✅ we'll create this next
]
