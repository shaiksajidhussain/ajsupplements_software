from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

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
