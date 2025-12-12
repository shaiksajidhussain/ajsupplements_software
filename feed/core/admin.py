from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User, Group
from django import forms
from django.utils import timezone
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
except ImportError:
    # Fallback for Python < 3.9
    from backports.zoneinfo import ZoneInfo
from .models import (
    Species,
    Subspecies,
    AnimalType,
    Phase,
    Ingredient,
    IngredientInclusionLimit,
    Premix,
    UserProfile,
    UserTaxonomyPermission,
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


# ---------- User Profile & Permissions ----------

class UserProfileInline(admin.StackedInline):
    model = UserProfile
    can_delete = False
    verbose_name_plural = "Profile"
    fields = ("is_trial", "trial_start_date", "trial_start_date_ist_display", "trial_end_date", "trial_end_date_ist_display", "is_active")
    readonly_fields = ("trial_start_date_ist_display", "trial_end_date_ist_display")

    @admin.display(description="Trial Start Date (IST)")
    def trial_start_date_ist_display(self, obj):
        if obj and obj.trial_start_date:
            ist_tz = ZoneInfo('Asia/Kolkata')
            ist_time = obj.trial_start_date.astimezone(ist_tz)
            return ist_time.strftime('%Y-%m-%d %H:%M:%S (IST)')
        return "-"
    
    @admin.display(description="Trial End Date (IST)")
    def trial_end_date_ist_display(self, obj):
        if obj and obj.trial_end_date:
            ist_tz = ZoneInfo('Asia/Kolkata')
            ist_time = obj.trial_end_date.astimezone(ist_tz)
            return ist_time.strftime('%Y-%m-%d %H:%M:%S (IST)')
        return "-"


class UserTaxonomyPermissionForm(forms.ModelForm):
    """Custom form with 'All' option in dropdowns"""
    
    class Meta:
        model = UserTaxonomyPermission
        fields = "__all__"
        widgets = {
            'species': forms.Select(attrs={'class': 'taxonomy-select'}),
            'subspecies': forms.Select(attrs={'class': 'taxonomy-select'}),
            'animal_type': forms.Select(attrs={'class': 'taxonomy-select'}),
            'phase': forms.Select(attrs={'class': 'taxonomy-select'}),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Make all fields optional
        self.fields['species'].required = False
        self.fields['subspecies'].required = False
        self.fields['animal_type'].required = False
        self.fields['phase'].required = False
        
        # Add "All" option at the top of each dropdown
        from .models import Species, Subspecies, AnimalType, Phase
        
        # Species dropdown with "All" option
        species_choices = [('', '--- All Species ---')]
        species_choices.extend([(s.id, s.name) for s in Species.objects.all().order_by('name')])
        self.fields['species'].widget.choices = species_choices
        
        # Subspecies dropdown with "All" option
        subspecies_choices = [('', '--- All Subspecies ---')]
        subspecies_choices.extend([(ss.id, f"{ss.species.name} - {ss.name}") for ss in Subspecies.objects.select_related('species').order_by('species__name', 'name')])
        self.fields['subspecies'].widget.choices = subspecies_choices
        
        # Animal Type dropdown with "All" option
        animal_type_choices = [('', '--- All Animal Types ---')]
        animal_type_choices.extend([(at.id, f"{at.subspecies.species.name} - {at.subspecies.name} - {at.name}") for at in AnimalType.objects.select_related('subspecies', 'subspecies__species').order_by('subspecies__species__name', 'subspecies__name', 'name')])
        self.fields['animal_type'].widget.choices = animal_type_choices
        
        # Phase dropdown with "All" option
        phase_choices = [('', '--- All Phases ---')]
        phase_choices.extend([(p.id, f"{p.animal_type.subspecies.species.name} - {p.animal_type.subspecies.name} - {p.animal_type.name} - {p.name}") for p in Phase.objects.select_related('animal_type', 'animal_type__subspecies', 'animal_type__subspecies__species').order_by('animal_type__subspecies__species__name', 'animal_type__subspecies__name', 'animal_type__name', 'name')])
        self.fields['phase'].widget.choices = phase_choices
        
        # Set initial value to empty string if None (to show "All" option selected)
        if self.instance and self.instance.pk:
            if self.instance.species is None:
                self.fields['species'].initial = ''
            if self.instance.subspecies is None:
                self.fields['subspecies'].initial = ''
            if self.instance.animal_type is None:
                self.fields['animal_type'].initial = ''
            if self.instance.phase is None:
                self.fields['phase'].initial = ''
    
    def clean(self):
        cleaned_data = super().clean()
        # Empty string means "All" - set to None
        if cleaned_data.get('species') == '':
            cleaned_data['species'] = None
        if cleaned_data.get('subspecies') == '':
            cleaned_data['subspecies'] = None
        if cleaned_data.get('animal_type') == '':
            cleaned_data['animal_type'] = None
        if cleaned_data.get('phase') == '':
            cleaned_data['phase'] = None
        return cleaned_data


class UserTaxonomyPermissionInline(admin.TabularInline):
    model = UserTaxonomyPermission
    form = UserTaxonomyPermissionForm
    extra = 1
    fields = ("species", "subspecies", "animal_type", "phase")
    verbose_name = "Taxonomy Permission"
    verbose_name_plural = "Taxonomy Permissions"


class CustomUserAdmin(BaseUserAdmin):
    inlines = (UserProfileInline, UserTaxonomyPermissionInline)
    list_display = ("username", "email", "first_name", "last_name", "is_staff", "profile_status")
    list_filter = ("is_staff", "is_superuser", "is_active", "profile__is_trial")
    
    # Hide Groups section from user form
    fieldsets = (
        (None, {"fields": ("username", "password")}),
        ("Personal info", {"fields": ("first_name", "last_name", "email")}),
        ("Permissions", {
            "fields": ("is_active", "is_staff", "is_superuser"),
        }),
        ("Important dates", {"fields": ("last_login", "date_joined")}),
    )
    
    add_fieldsets = (
        (None, {
            "classes": ("wide",),
            "fields": ("username", "password1", "password2"),
        }),
    )

    @admin.display(description="Status")
    def profile_status(self, obj):
        if hasattr(obj, 'profile'):
            profile = obj.profile
            if profile.is_trial:
                if profile.is_trial_expired:
                    return "Trial (Expired)"
                return "Trial"
            return "Paid"
        return "No Profile"
    profile_status.short_description = "Status"

    def save_formset(self, request, form, formset, change):
        """Override to handle UserProfile inline - update existing instead of creating new"""
        if formset.model == UserProfile:
            # Get or create profile for the user (signal may have already created it)
            profile, created = UserProfile.objects.get_or_create(
                user=form.instance,
                defaults={'is_trial': True}
            )
            
            # Set form instances to existing profile so Django updates instead of creates
            for inline_form in formset.forms:
                if inline_form.cleaned_data and not inline_form.cleaned_data.get('DELETE', False):
                    # Point form instance to existing profile
                    inline_form.instance = profile
                    inline_form.instance.user = form.instance
                    # Update fields from form data
                    if 'is_trial' in inline_form.cleaned_data:
                        inline_form.instance.is_trial = inline_form.cleaned_data['is_trial']
                    if 'trial_start_date' in inline_form.cleaned_data:
                        inline_form.instance.trial_start_date = inline_form.cleaned_data['trial_start_date']
                    if 'trial_end_date' in inline_form.cleaned_data:
                        inline_form.instance.trial_end_date = inline_form.cleaned_data['trial_end_date']
                    if 'is_active' in inline_form.cleaned_data:
                        inline_form.instance.is_active = inline_form.cleaned_data['is_active']
            
            # Now call parent to save (it will update the existing profile, not create new)
            super().save_formset(request, form, formset, change)
        else:
            # For other formsets (like UserTaxonomyPermission), use default behavior
            super().save_formset(request, form, formset, change)


# Hide Groups from admin sidebar
admin.site.unregister(Group)

# Re-register UserAdmin
admin.site.unregister(User)
admin.site.register(User, CustomUserAdmin)


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "is_trial", "trial_start_date", "trial_end_date", "is_active", "trial_status")
    list_filter = ("is_trial", "is_active", "trial_start_date")
    search_fields = ("user__username", "user__email")
    readonly_fields = ("created_at", "updated_at")
    fieldsets = (
        ("User", {
            "fields": ("user",)
        }),
        ("Trial Information", {
            "fields": ("is_trial", "trial_start_date", "trial_end_date")
        }),
        ("Status", {
            "fields": ("is_active",)
        }),
        ("Timestamps", {
            "fields": ("created_at", "updated_at"),
            "classes": ("collapse",)
        }),
    )

    @admin.display(description="Trial Status")
    def trial_status(self, obj):
        if not obj.is_trial:
            return "Paid User"
        if obj.is_trial_expired:
            return "Expired"
        if obj.trial_end_date:
            return f"Active (ends {obj.trial_end_date.strftime('%Y-%m-%d')})"
        return "Active (Unlimited)"


@admin.register(UserTaxonomyPermission)
class UserTaxonomyPermissionAdmin(admin.ModelAdmin):
    list_display = ("user", "species", "subspecies", "animal_type", "phase", "created_at")
    list_filter = ("species", "subspecies", "animal_type", "phase", "created_at")
    search_fields = ("user__username", "user__email", "species__name", "subspecies__name", 
                     "animal_type__name", "phase__name")
    autocomplete_fields = ("user", "species", "subspecies", "animal_type", "phase")
    ordering = ("user__username", "species__name", "subspecies__name", "animal_type__name", "phase__name")
