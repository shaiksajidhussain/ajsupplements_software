from django.contrib import admin
from .models import (
    Species,
    Subspecies,
    AnimalType,
    Phase,
    Ingredient,
    IngredientInclusionLimit,
    Premix,
)

# ---------- Inlines ----------

class InclusionLimitForPhaseInline(admin.TabularInline):
    model = IngredientInclusionLimit
    fk_name = "phase"
    extra = 0
    autocomplete_fields = ["ingredient", "species", "subspecies", "animal_type"]
    fields = ("species", "subspecies", "animal_type", "ingredient", "max_inclusion")
    show_change_link = True


class InclusionLimitForIngredientInline(admin.TabularInline):
    model = IngredientInclusionLimit
    fk_name = "ingredient"
    extra = 0
    autocomplete_fields = ["species", "subspecies", "animal_type", "phase"]
    fields = ("species", "subspecies", "animal_type", "phase", "max_inclusion")
    show_change_link = True


# ---------- Species / Subspecies ----------

@admin.register(Species)
class SpeciesAdmin(admin.ModelAdmin):
    list_display = ("name",)
    search_fields = ("name",)
    ordering = ("name",)


@admin.register(Subspecies)
class SubspeciesAdmin(admin.ModelAdmin):
    list_display = ("name", "species")
    list_filter = ("species",)
    search_fields = ("name", "species__name")
    ordering = ("species__name", "name")
    autocomplete_fields = ("species",)


# ---------- AnimalType (show Species; add read-only Species on form) ----------

@admin.register(AnimalType)
class AnimalTypeAdmin(admin.ModelAdmin):
    list_display = ("name", "subspecies", "species_display")
    list_filter = ("subspecies__species", "subspecies")
    search_fields = ("name", "subspecies__name", "subspecies__species__name")
    ordering = ("subspecies__species__name", "subspecies__name", "name")
    autocomplete_fields = ("subspecies",)
    readonly_fields = ("species_display",)

    fieldsets = (
        (None, {
            "fields": ("name", "subspecies", "species_display")
        }),
    )

    @admin.display(description="Species")
    def species_display(self, obj):
        return obj.subspecies.species.name


# ---------- Phase (show Species; group nutrient requirements) ----------

@admin.register(Phase)
class PhaseAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "animal_type",
        "subspecies_display",
        "species_display",
        # Key nutrients in list (so you can scan requirements quickly)
        "crude_protein",
        "energy",
        "calcium",
        "phosphorus",
        "lysine",
        "methionine",
        "salt",
        "crude_fiber",
    )
    list_filter = (
        "animal_type__subspecies__species",
        "animal_type__subspecies",
        "animal_type",
    )
    search_fields = (
        "name",
        "animal_type__name",
        "animal_type__subspecies__name",
        "animal_type__subspecies__species__name",
    )
    ordering = (
        "animal_type__subspecies__species__name",
        "animal_type__subspecies__name",
        "animal_type__name",
        "name",
    )
    autocomplete_fields = ("animal_type",)
    readonly_fields = ("species_display", "subspecies_display")

    # Group nutrients clearly on the form
    fieldsets = (
        ("Basic", {
            "fields": ("name", "animal_type", "species_display", "subspecies_display")
        }),
        ("Nutrient Requirements (Targets)", {
            "fields": (
                "energy",
                "crude_protein",
                "lysine",
                "methionine",
                "calcium",
                "phosphorus",
                "salt",
                "crude_fiber",
            ),
            "description": "Set phase-specific requirement targets. Energy is kcal/kg; others are in %."
        }),
    )

    inlines = [InclusionLimitForPhaseInline]

    @admin.display(description="Subspecies")
    def subspecies_display(self, obj):
        return obj.animal_type.subspecies.name

    @admin.display(description="Species")
    def species_display(self, obj):
        return obj.animal_type.subspecies.species.name


# ---------- Ingredient ----------

@admin.register(Ingredient)
class IngredientAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "category",
        "price_per_kg",
        "energy",
        "crude_protein",
        "calcium",
        "phosphorus",
        "lysine",
        "methionine",
        "salt",
        "crude_fiber",
    )
    list_filter = ("category",)
    search_fields = ("name",)
    ordering = ("name",)
    inlines = [InclusionLimitForIngredientInline]


# ---------- Inclusion Limit (standalone) ----------

@admin.register(IngredientInclusionLimit)
class IngredientInclusionLimitAdmin(admin.ModelAdmin):
    list_display = (
        "ingredient",
        "max_inclusion",
        "phase",
        "animal_type",
        "subspecies",
        "species",
    )
    list_filter = (
        "species",
        "subspecies",
        "animal_type",
        "phase",
        "ingredient__category",
        "ingredient",
    )
    search_fields = (
        "ingredient__name",
        "phase__name",
        "animal_type__name",
        "subspecies__name",
        "species__name",
    )
    ordering = (
        "species__name",
        "subspecies__name",
        "animal_type__name",
        "phase__name",
        "ingredient__name",
    )
    autocomplete_fields = ("species", "subspecies", "animal_type", "phase", "ingredient")


# ---------- Premix (simple) ----------

@admin.register(Premix)
class PremixAdmin(admin.ModelAdmin):
    list_display = ("name", "inclusion_rate", "price_per_kg", "phase", "animal_type", "subspecies", "species")
    list_filter = ("species", "subspecies", "animal_type", "phase")
    search_fields = ("name", "species__name", "subspecies__name", "animal_type__name", "phase__name")
    ordering = ("species__name", "name")
    autocomplete_fields = ("species", "subspecies", "animal_type", "phase")
