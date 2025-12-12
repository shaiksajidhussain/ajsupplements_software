from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django.contrib.auth.models import User
from django.utils import timezone
from django.db.models.signals import post_save
from django.dispatch import receiver

# ---------- Taxonomy ----------
class Species(models.Model):
    name = models.CharField(max_length=100, unique=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name


class Subspecies(models.Model):
    species = models.ForeignKey(
        Species, on_delete=models.CASCADE, related_name="subspecies"
    )
    name = models.CharField(max_length=100)

    class Meta:
        unique_together = ("species", "name")
        ordering = ["species__name", "name"]

    def __str__(self):
        return f"{self.species.name} - {self.name}"


class AnimalType(models.Model):
    subspecies = models.ForeignKey(
        Subspecies, on_delete=models.CASCADE, related_name="animal_types"
    )
    name = models.CharField(max_length=100)

    class Meta:
        unique_together = ("subspecies", "name")
        ordering = ["subspecies__species__name", "subspecies__name", "name"]

    def __str__(self):
        return f"{self.subspecies.name} - {self.name}"


class Phase(models.Model):
    animal_type = models.ForeignKey(
        AnimalType, on_delete=models.CASCADE, related_name="phases"
    )
    name = models.CharField(max_length=100)

    # Targets
    crude_protein = models.FloatField(help_text="In %")
    energy = models.FloatField(help_text="In kcal/kg")
    calcium = models.FloatField(help_text="In %")
    phosphorus = models.FloatField(help_text="In %")
    lysine = models.FloatField(help_text="In %")
    methionine = models.FloatField(help_text="In %")
    salt = models.FloatField(help_text="In %")
    crude_fiber = models.FloatField(default=0.0, help_text="Maximum crude fiber allowed (%)")

    class Meta:
        unique_together = ("animal_type", "name")
        ordering = [
            "animal_type__subspecies__species__name",
            "animal_type__subspecies__name",
            "animal_type__name",
            "name",
        ]

    def __str__(self):
        return f"{self.animal_type.name} - {self.name}"


# ---------- Ingredient ----------
class Ingredient(models.Model):
    CATEGORY_CHOICES = [
        ("protein", "Protein Source"),
        ("medium", "Medium Source"),
        ("energy", "Energy Source"),
        ("mineral", "Mineral"),
    ]

    name = models.CharField(max_length=100, unique=True)
    category = models.CharField(max_length=10, choices=CATEGORY_CHOICES)

    # Macronutrient composition (per % inclusion, consistent with your solver)
    crude_protein = models.FloatField(default=0.0, help_text="In %")
    energy = models.FloatField(default=0.0, help_text="kcal/kg")
    calcium = models.FloatField(default=0.0)
    phosphorus = models.FloatField(default=0.0)
    lysine = models.FloatField(default=0.0)
    methionine = models.FloatField(default=0.0)
    salt = models.FloatField(default=0.0)
    crude_fiber = models.FloatField(default=0.0, help_text="In %")

    # Cost
    price_per_kg = models.FloatField(default=0.0, help_text="Cost per kg (₹)")

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name


# ---------- Inclusion limits (now includes Phase) ----------
class IngredientInclusionLimit(models.Model):
    species = models.ForeignKey(Species, on_delete=models.CASCADE)
    subspecies = models.ForeignKey(Subspecies, on_delete=models.CASCADE)
    animal_type = models.ForeignKey(AnimalType, on_delete=models.CASCADE)
    phase = models.ForeignKey(Phase, on_delete=models.CASCADE)
    ingredient = models.ForeignKey(Ingredient, on_delete=models.CASCADE)

    max_inclusion = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
        help_text="Max % for this ingredient in the specified taxonomy + phase"
    )

    class Meta:
        unique_together = ("species", "subspecies", "animal_type", "phase", "ingredient")
        ordering = [
            "species__name",
            "subspecies__name",
            "animal_type__name",
            "phase__name",
            "ingredient__name",
        ]

    def __str__(self):
        return (
            f"{self.species.name} - {self.subspecies.name} - {self.animal_type.name} - "
            f"{self.phase.name} - {self.ingredient.name} → Max {self.max_inclusion}%"
        )


# ---------- Premix (simple; no micronutrients) ----------
class Premix(models.Model):
    name = models.CharField(max_length=100)
    species = models.ForeignKey(Species, on_delete=models.CASCADE)
    subspecies = models.ForeignKey(Subspecies, on_delete=models.CASCADE, null=True, blank=True)
    animal_type = models.ForeignKey(AnimalType, on_delete=models.CASCADE, null=True, blank=True)
    phase = models.ForeignKey(Phase, on_delete=models.CASCADE, null=True, blank=True)

    price_per_kg = models.FloatField(default=0.0)
    inclusion_rate = models.FloatField(
        default=0.25,
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        help_text="% inclusion in feed (e.g. 0.25 for 0.25%)"
    )

    class Meta:
        unique_together = ("species", "subspecies", "animal_type", "phase")
        ordering = ["species__name", "name"]

    def __str__(self):
        scope = []
        scope.append(self.species.name)
        if self.subspecies_id:
            scope.append(self.subspecies.name)
        if self.animal_type_id:
            scope.append(self.animal_type.name)
        if self.phase_id:
            scope.append(self.phase.name)
        return f"{self.name} ({' / '.join(scope)})"


# ---------- User Profile & Permissions ----------
class UserProfile(models.Model):
    """Extended user profile with trial information"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    is_trial = models.BooleanField(default=True, help_text="Is this a trial user?")
    trial_start_date = models.DateTimeField(default=timezone.now, help_text="When trial started")
    trial_end_date = models.DateTimeField(null=True, blank=True, help_text="When trial ends (null = unlimited)")
    is_active = models.BooleanField(default=True, help_text="Is user account active?")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        status = "Trial" if self.is_trial else "Paid"
        return f"{self.user.username} ({status})"

    @property
    def is_trial_expired(self):
        """Check if trial has expired"""
        if not self.is_trial:
            return False
        if self.trial_end_date is None:
            return False
        return timezone.now() > self.trial_end_date


class UserTaxonomyPermission(models.Model):
    """Controls which taxonomy items (species, subspecies, animal types, phases) a user can access"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='taxonomy_permissions')
    
    # Allow None for "all" - if None, user can access all items of that type
    species = models.ForeignKey(Species, on_delete=models.CASCADE, null=True, blank=True, 
                                 help_text="None = all species allowed")
    subspecies = models.ForeignKey(Subspecies, on_delete=models.CASCADE, null=True, blank=True,
                                   help_text="None = all subspecies allowed (if species is set, only within that species)")
    animal_type = models.ForeignKey(AnimalType, on_delete=models.CASCADE, null=True, blank=True,
                                    help_text="None = all animal types allowed")
    phase = models.ForeignKey(Phase, on_delete=models.CASCADE, null=True, blank=True,
                             help_text="None = all phases allowed")
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("user", "species", "subspecies", "animal_type", "phase")
        ordering = ['user__username', 'species__name', 'subspecies__name', 'animal_type__name', 'phase__name']

    def __str__(self):
        parts = []
        if self.species:
            parts.append(f"Species: {self.species.name}")
        if self.subspecies:
            parts.append(f"Subspecies: {self.subspecies.name}")
        if self.animal_type:
            parts.append(f"AnimalType: {self.animal_type.name}")
        if self.phase:
            parts.append(f"Phase: {self.phase.name}")
        if not parts:
            parts.append("All")
        return f"{self.user.username} - {' / '.join(parts)}"


# Signals to auto-create UserProfile
@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    """Automatically create UserProfile when a User is created"""
    if created:
        UserProfile.objects.get_or_create(user=instance, defaults={'is_trial': True})
