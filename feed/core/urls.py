from django.urls import path
from . import views

urlpatterns = [
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("", views.home, name="home"),
    path("solve/", views.solve_feed_formula, name="solve_feed_formula"),
    # If/when you add a real ingredient list view in views.py, uncomment the next line:
    # path("ingredients/", views.ingredient_list, name="ingredient_list"),
]
