# Opened Files
## File Name
feed/core/views.py
## File Content
from __future__ import annotations

from typing import Dict, List, Any

from django.http import JsonResponse, HttpResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods, require_POST
from django.contrib import messages
import json
import io

# Optional imports for PDF/Excel export
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

from .models import (
    Species, Subspecies, AnimalType, Phase,
    Ingredient, Premix, IngredientInclusionLimit,
)

try:
    import pulp as pl
except Exception:
    pl = None


# -------------------------
# Helpers
# -------------------------

def _ingredients_by_category_qs():
    by_cat = {"protein": [], "medium": [], "energy": []}
    qs = Ingredient.objects.all().only("id", "name", "category").order_by("name")
    for ing in qs:
        if ing.category in by_cat:
            by_cat[ing.category].append(ing)
    return by_cat


def _inclusion_limits_map(species, subspecies, animal_type, phase) -> Dict[int, float]:
    lims: Dict[int, float] = {}
    qs = IngredientInclusionLimit.objects.filter(
        species=species, subspecies=subspecies, animal_type=animal_type, phase=phase
    ).only("ingredient_id", "max_inclusion")
    for row in qs:
        lims[row.ingredient_id] = float(row.max_inclusion)
    return lims


def _row_from_ing(ing: Ingredient, limits: Dict[int, float], default_max: float = 100.0) -> Dict[str, Any]:
    return {
        "id": ing.id,
        "name": ing.name,
        "category": ing.category,  # protein | medium | energy | mineral
        "cost": float(ing.price_per_kg or 0.0),
        "ME": float(ing.energy or 0.0),            # kcal/kg
        "CP": float(ing.crude_protein or 0.0),     # %
        "Lys": float(ing.lysine or 0.0),           # %
        "Met": float(ing.methionine or 0.0),       # %
        "Ca": float(ing.calcium or 0.0),           # %
        "P": float(ing.phosphorus or 0.0),         # %
        "NaCl": float(ing.salt or 0.0),            # %
        "CF": float(ing.crude_fiber or 0.0),       # %
        "min": 0.0,
        "max": float(limits.get(ing.id, default_max)),
    }


# -------------------------
# Authentication Views
# -------------------------

@require_http_methods(["GET", "POST"])
def login_view(request):
    """Login page - redirects to home if already logged in"""
    if request.user.is_authenticated:
        return redirect('home')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        if username and password:
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f'Welcome back, {user.username}!')
                next_url = request.GET.get('next', 'home')
                return redirect(next_url)
            else:
                messages.error(request, 'Invalid username or password.')
        else:
            messages.error(request, 'Please provide both username and password.')
    
    return render(request, 'core/login.html')


@login_required
@require_http_methods(["POST", "GET"])
def logout_view(request):
    """Logout view"""
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('login')


# -------------------------
# Application Views
# -------------------------

@login_required
@require_http_methods(["GET"])
def home(request):
    context = {
        "species": Species.objects.all().order_by("name"),
        "subspecies": Subspecies.objects.select_related("species").order_by("species__name", "name"),
        "animal_types": AnimalType.objects.select_related("subspecies", "subspecies__species")
            .order_by("subspecies__species__name", "subspecies__name", "name"),
        "phases": Phase.objects.select_related(
            "animal_type", "animal_type__subspecies", "animal_type__subspecies__species"
        ).order_by(
            "animal_type__subspecies__species__name",
            "animal_type__subspecies__name",
            "animal_type__name", "name",
        ),
        "premixes": Premix.objects.select_related("species", "subspecies", "animal_type", "phase")
            .order_by("species__name", "name"),
        "ingredients_by_category": _ingredients_by_category_qs(),
    }
    return render(request, "home.html", context)


@login_required
@require_POST
def solve_feed_formula(request):
    """
    LP + 5-pass overshoot trim + final energy & protein rebalancing (NO top-up):

    - Robust mins and CF guard in LP, with minerals reserved cap
    - Auto-add oil only if energy is selected
    - Minerals are NOT used as energy sources in constraints/trim (ME ignored for them),
      but their ME is still visible in the UI report.
    - 5 iterations of safe trim:
        * CP surplus -> reduce proteins
        * ME surplus -> reduce oil, then other energy
        * Minerals: if Ca & P both high -> trim duals (e.g., DCP), else trim single sources
      For every cut, add the SAME % to a filler while guaranteeing all mins stay >= targets.
    - Final rebalancing:
        * redistribute_energy_surplus(): if ME is comfortably above target,
          trade high-ME expensive donors for cheaper, lower-ME fillers.
        * final_protein_trim(): if CP > CP_min + 0.02, trim high-cost, high-CP protein donors
          and move % into cheap fillers (energy → medium → protein → oil).
    - No top-up phase at the end.
    """
    if pl is None:
        return JsonResponse(
            {"status": "error", "message": "PuLP is not installed. Run: pip install pulp"},
            status=500,
        )

    # ---- Taxonomy selections
    try:
        species_id = int(request.POST.get("species_id"))
        subspecies_id = int(request.POST.get("subspecies_id"))
        animal_type_id = int(request.POST.get("animal_type_id"))
        phase_id = int(request.POST.get("phase_id"))
    except (TypeError, ValueError):
        return JsonResponse(
            {"status": "error", "message": "Please select species, subspecies, animal type, and phase."},
            status=400,
        )

    species = get_object_or_404(Species, pk=species_id)
    subspecies = get_object_or_404(Subspecies, pk=subspecies_id, species=species)
    animal_type = get_object_or_404(AnimalType, pk=animal_type_id, subspecies=subspecies)
    phase = get_object_or_404(Phase, pk=phase_id, animal_type=animal_type)

    # ---- Phase targets
    targets = {
        "ME_min": float(phase.energy or 0.0),
        "CP_min": float(phase.crude_protein or 0.0),
        "Lys_min": float(phase.lysine or 0.0),
        "Met_min": float(phase.methionine or 0.0),
        "Ca_min": float(phase.calcium or 0.0),
        "P_min": float(phase.phosphorus or 0.0),
        "NaCl_min": float(phase.salt or 0.0),
        "CF_max": float(phase.crude_fiber or 0.0),
    }

    # Robust safety factors (relative potency deltas) used only in LP
    # NOTE: NaCl robustness = 0.0 to avoid infeasible cases when target == cap.
    robust = {
        "ME": 0.02,
        "CP": 0.02,
        "Lys": 0.01,
        "Met": 0.01,
        "Ca": 0.01,
        "P": 0.01,
        "NaCl": 0.0,
        "CF": 0.015,
    }

    # ---- Premix (optional)
    use_premix = request.POST.get("use_premix") == "on"
    premix_row = None
    if use_premix:
        premix_id = request.POST.get("premix_id")
        if not premix_id:
            return JsonResponse(
                {"status": "error", "message": "Premix checked but no premix selected."},
                status=400,
            )
        pmx = get_object_or_404(Premix, pk=premix_id)
        
        # Get premix percentage from form, or use default from model
        premix_percentage = request.POST.get("premix_percentage")
        if premix_percentage and premix_percentage.strip():
            try:
                inclusion_rate = float(premix_percentage)
                if inclusion_rate < 0 or inclusion_rate > 100:
                    return JsonResponse(
                        {"status": "error", "message": "Premix percentage must be between 0 and 100."},
                        status=400,
                    )
            except ValueError:
                return JsonResponse(
                    {"status": "error", "message": "Invalid premix percentage value."},
                    status=400,
                )
        else:
            # Use default inclusion rate from model
            inclusion_rate = float(pmx.inclusion_rate or 0.0)
        
        premix_row = {
            "id": f"premix:{pmx.id}",
            "name": pmx.name,
            "category": "premix",
            "cost": float(pmx.price_per_kg or 0.0),
            "ME": 0.0, "CP": 0.0, "Lys": 0.0, "Met": 0.0,
            "Ca": 0.0, "P": 0.0, "NaCl": 0.0, "CF": 0.0,
            "min": inclusion_rate,
            "max": inclusion_rate,
        }

    # ---- User ingredients
    raw_ids = request.POST.getlist("ingredients")
    try:
        selected_ids = [int(i) for i in raw_ids]
    except ValueError:
        return JsonResponse({"status": "error", "message": "Invalid ingredient IDs."}, status=400)
    if not selected_ids:
        return JsonResponse({"status": "error", "message": "Please select at least one ingredient."}, status=400)

    selected_qs = list(Ingredient.objects.filter(id__in=selected_ids).order_by("name"))
    selected_medium = [ing for ing in selected_qs if ing.category == "medium"]
    selected_protein = [ing for ing in selected_qs if ing.category == "protein"]
    selected_energy = [ing for ing in selected_qs if ing.category == "energy"]

    # ---- Limits
    lim_map = _inclusion_limits_map(species, subspecies, animal_type, phase)

    # ---------------------------
    # Slack policy
    # ---------------------------
    total_space = 100.0
    has_medium = len(selected_medium) > 0
    slack_target = 20.0 if has_medium else 15.0

    medium_rows: List[Dict[str, Any]] = []
    medium_fixed_sum = 0.0

    if len(selected_medium) == 1:
        m = selected_medium[0]
        limit = float(lim_map.get(m.id, 0.0))
        if limit > 0:
            row = _row_from_ing(m, lim_map, default_max=limit)
            row["min"] = min(limit, row["max"])
            row["max"] = row["min"]
            medium_rows.append(row)
            medium_fixed_sum = row["min"]
    elif len(selected_medium) > 1:
        for m in selected_medium:
            row = _row_from_ing(m, lim_map, default_max=100.0)
            row["min"] = min(5.0, row["max"])
            medium_rows.append(row)
            medium_fixed_sum += row["min"]

    premix_fixed = float(premix_row["min"]) if premix_row else 0.0
    minerals_reserved_cap = max(slack_target - (medium_fixed_sum + premix_fixed), 0.0)

    # ---------------------------
    # Build rows for LP
    # ---------------------------
    rows: List[Dict[str, Any]] = []
    rows.extend(medium_rows)
    if premix_row:
        rows.append(premix_row)

    for ing in selected_protein:
        rows.append(_row_from_ing(ing, lim_map, default_max=100.0))
    for ing in selected_energy:
        rows.append(_row_from_ing(ing, lim_map, default_max=100.0))

    # Auto oil if any energy is selected
    oil_added = False
    oil_id: Any = None
    if selected_energy:
        try:
            oil_price = float(request.POST.get("oil_price_per_kg", "0") or 0.0)
        except ValueError:
            oil_price = 0.0
        try:
            oil_cap_val = request.POST.get("oil_cap_percent")
            oil_cap = float(oil_cap_val) if oil_cap_val not in (None, "",) else 3.0
        except ValueError:
            oil_cap = 3.0

        oil_row = {
            "id": "oil:7000",
            "name": "Oil (7000 kcal/kg)",
            "category": "energy",
            "cost": oil_price,
            "ME": 9000.0,
            "CP": 0.0, "Lys": 0.0, "Met": 0.0,
            "Ca": 0.0, "P": 0.0, "NaCl": 0.0, "CF": 0.0,
            "min": 0.0,
            "max": float(oil_cap if oil_cap is not None else 3.0),
            "is_oil": True,
        }
        rows.append(oil_row)
        oil_added = True
        oil_id = oil_row["id"]

    # Minerals (auto) with defensive 5% max (unless phase limit present)
    minerals_qs = list(Ingredient.objects.filter(category="mineral").order_by("name"))
    for ing in minerals_qs:
        rows.append(_row_from_ing(ing, lim_map, default_max=5.0))

    # --------------------------------------------------------------------------------
    # Helper: nutrient value used in constraints (ignore mineral ME for ME constraint)
    # --------------------------------------------------------------------------------
    def _eff_val(row: Dict[str, Any], k: str) -> float:
        # For ME: DO NOT count minerals as energy source in constraints
        if k == "ME" and row["category"] == "mineral":
            return 0.0
        return row[k]

    # ---------------------------
    # LP Level 1: feasibility (robust)
    # ---------------------------
    mins = ["ME", "CP", "Lys", "Met", "Ca", "P", "NaCl"]

    m1 = pl.LpProblem("feasibility", pl.LpMinimize)
    x = {
        row["id"]: pl.LpVariable(
            f"x_{str(row['id']).replace(':','_')}",
            lowBound=row["min"],
            upBound=row["max"],
        )
        for row in rows
    }
    m1 += pl.lpSum([x[i] for i in x]) == total_space

    u = {k: pl.LpVariable(f"u_{k}", lowBound=0) for k in mins}
    for k in mins:
        r = robust.get(k, 0.0)
        m1 += (
            pl.lpSum([(_eff_val(row, k) * (1 - r)) * x[row["id"]] for row in rows]) / 100.0
            + u[k]
            >= targets[f"{k}_min"]
        )

    rcf = robust.get("CF", 0.0)
    v_cf = pl.LpVariable("v_cf", lowBound=0)
    m1 += (
        pl.lpSum([(row["CF"] * (1 + rcf)) * x[row["id"]] for row in rows]) / 100.0
        - v_cf
        <= targets["CF_max"]
    )

    mineral_ids_all = [row["id"] for row in rows if row["category"] == "mineral"]
    if mineral_ids_all and minerals_reserved_cap is not None:
        m1 += pl.lpSum([x[i] for i in mineral_ids_all]) <= minerals_reserved_cap

    BIG = 1e6
    m1 += pl.lpSum([BIG * u[k] for k in u]) + BIG * v_cf
    m1.solve(pl.PULP_CBC_CMD(msg=False))

    if any((u[k].value() or 0.0) > 1e-7 for k in u) or (v_cf.value() or 0.0) > 1e-7:
        # ------------------ DEBUG START ------------------
        print("========== FEASIBILITY DEBUG ==========")
        print("Targets:", targets)
        print(f"Minerals reserved cap: {minerals_reserved_cap}")
        print("Mineral rows:")
        for row in rows:
            if row["category"] == "mineral":
                print(
                    f"  {row['name']}: "
                    f"Ca={row['Ca']} P={row['P']} Lys={row['Lys']} "
                    f"Met={row['Met']} NaCl={row['NaCl']} "
                    f"min={row['min']} max={row['max']}"
                )

        # Left-hand side of each nutrient constraint (with robust deltas)
        for k in mins:
            r = robust.get(k, 0.0)
            lhs = sum(
                (_eff_val(row, k) * (1 - r)) * (x[row["id"]].value() or 0.0) / 100.0
                for row in rows
            )
            print(
                f"{k}: "
                f"lhs={lhs:.4f}, "
                f"target_min={targets[f'{k}_min']}, "
                f"slack={float(u[k].value() or 0.0):.6f}"
            )

        # Fiber constraint LHS
        cf_lhs = sum(
            (row["CF"] * (1 + rcf)) * (x[row["id"]].value() or 0.0) / 100.0
            for row in rows
        )
        print(
            f"CF: lhs={cf_lhs:.4f}, "
            f"target_max={targets['CF_max']}, "
            f"slack={float(v_cf.value() or 0.0):.6f}"
        )

        # Minerals total usage vs cap
        if mineral_ids_all:
            used_minerals = sum((x[i].value() or 0.0) for i in mineral_ids_all)
            print(
                f"Mineral usage: {used_minerals:.4f} % "
                f"vs cap {minerals_reserved_cap:.4f} %"
            )

        print("============== END DEBUG ==============")
        # ------------------ DEBUG END --------------------

        binding = {k: float(u[k].value() or 0.0) for k in u}
        return JsonResponse({
            "status": "error",
            "message": (
                "Infeasible under current limits/targets. "
                "Try raising caps, adding a protein/energy source, or relaxing fiber."
            ),
            "binding_slacks": binding,
            "fiber_slack": float(v_cf.value() or 0.0),
        }, status=422)

    # ---------------------------
    # LP Level 2: least cost (robust)
    # ---------------------------
    m2 = pl.LpProblem("least_cost", pl.LpMinimize)
    y = {
        row["id"]: pl.LpVariable(
            f"y_{str(row['id']).replace(':','_')}",
            lowBound=row["min"],
            upBound=row["max"],
        )
        for row in rows
    }
    m2 += pl.lpSum([y[i] for i in y]) == total_space

    for k in mins:
        r = robust.get(k, 0.0)
        m2 += (
            pl.lpSum([(_eff_val(row, k) * (1 - r)) * y[row["id"]] for row in rows]) / 100.0
            >= targets[f"{k}_min"]
        )

    m2 += (
        pl.lpSum([(row["CF"] * (1 + rcf)) * y[row["id"]] for row in rows]) / 100.0
        <= targets["CF_max"]
    )

    if mineral_ids_all and minerals_reserved_cap is not None:
        m2 += pl.lpSum([y[i] for i in mineral_ids_all]) <= minerals_reserved_cap

    m2 += pl.lpSum([row["cost"] * y[row["id"]] for row in rows])
    m2.solve(pl.PULP_CBC_CMD(msg=False))

    sol = {i: float(y[i].value() or 0.0) for i in y}

    # ---------------------------
    # 5-pass overshoot trim (NO top-up)
    # ---------------------------
    id2row = {r["id"]: r for r in rows}
    protein_ids = [r["id"] for r in rows if r["category"] == "protein"]
    energy_ids = [r["id"] for r in rows if r["category"] == "energy"]
    non_oil_energy = [i for i in energy_ids if i != oil_id]
    mineral_ids = [r["id"] for r in rows if r["category"] == "mineral"]
    medium_ids = [r["id"] for r in rows if r["category"] == "medium"]

    def profile(s: Dict[Any, float]) -> Dict[str, float]:
        """
        Internal profile for trimming:
        - ME ignores minerals as energy source.
        - All other nutrients counted as normal.
        """
        def val(k: str) -> float:
            total = 0.0
            for iid, pct in s.items():
                v = id2row[iid][k]
                if k == "ME" and id2row[iid]["category"] == "mineral":
                    v = 0.0
                total += v * pct / 100.0
            return total

        return {
            "total": sum(s.values()),
            "ME": val("ME"), "CP": val("CP"),
            "Ca": val("Ca"), "P": val("P"),
            "Lys": val("Lys"), "Met": val("Met"),
            "NaCl": val("NaCl"), "CF": val("CF"),
        }

    def headroom_down(i: Any) -> float:
        return sol.get(i, 0.0) - id2row[i]["min"]

    def headroom_up(i: Any) -> float:
        return id2row[i]["max"] - sol.get(i, 0.0)

    # filler: prefer non-oil energy, then oil, then any cheapest with headroom
    def pick_filler_for(nkey: str) -> Any | None:
        pool = non_oil_energy[:] if non_oil_energy else []
        if oil_id:
            pool.append(oil_id)
        if not pool:
            pool = [rid for rid in id2row]
        pool = [rid for rid in pool if headroom_up(rid) > 1e-9]
        if not pool:
            return None
        pool.sort(key=lambda rid: (id2row[rid]["cost"], -id2row[rid].get(nkey, 0.0)))
        return pool[0]

    def safe_cut_with_fill(iid: Any, fid: Any, nkey: str, max_want: float) -> float:
        """
        Compute safe cut on iid while adding SAME % to filler fid,
        making sure ALL mins remain >= targets after the change and CF <= CF_max.
        Returns the cut we can actually take.

        Note: for ME checks, minerals' ME is treated as 0 (no energy source).
        """
        if max_want <= 1e-9:
            return 0.0
        cur = sol.get(iid, 0.0)
        if cur <= 1e-9:
            return 0.0
        cut_cap = min(max_want, headroom_down(iid), headroom_up(fid))
        if cut_cap <= 1e-9:
            return 0.0

        # Limit by each min nutrient
        dens_i: Dict[str, float] = {}
        dens_f: Dict[str, float] = {}
        for k in ["ME", "CP", "Lys", "Met", "Ca", "P", "NaCl"]:
            vi = id2row[iid][k]
            vf = id2row[fid][k]
            if k == "ME":
                if id2row[iid]["category"] == "mineral":
                    vi = 0.0
                if id2row[fid]["category"] == "mineral":
                    vf = 0.0
            dens_i[k] = vi / 100.0
            dens_f[k] = vf / 100.0

        prof = profile(sol)
        max_cut_all = cut_cap
        for k in ["ME", "CP", "Lys", "Met", "Ca", "P", "NaCl"]:
            surplus_k = prof[k] - targets[f"{k}_min"]
            delta_per = dens_f[k] - dens_i[k]  # net change per 1% move
            if delta_per < 0:  # this move reduces nutrient k
                limit_k = surplus_k / (-delta_per) if surplus_k > 0 else 0.0
                max_cut_all = min(max_cut_all, max(0.0, limit_k))

        # CF guard (1% move changes CF by dens_f_CF - dens_i_CF)
        dens_i_cf = id2row[iid]["CF"] / 100.0
        dens_f_cf = id2row[fid]["CF"] / 100.0
        delta_cf = dens_f_cf - dens_i_cf
        if delta_cf > 0:
            allow_cf = max(0.0, targets["CF_max"] - prof["CF"])
            max_cf_cut = allow_cf / delta_cf if delta_cf > 0 else cut_cap
            max_cut_all = min(max_cut_all, max_cf_cut)

        return max(0.0, min(cut_cap, max_cut_all))

    def trim_group(nkey: str, candidate_ids: List[Any]) -> float:
        """
        Trim surplus of nkey using candidates; for each cut add SAME % to filler chosen for nkey.
        Returns total % cut (and equally added as filler).
        """
        prof0 = profile(sol)
        surplus0 = prof0[nkey] - targets[f"{nkey}_min"]
        if surplus0 <= 1e-9 or not candidate_ids:
            return 0.0

        # order: proteins by high cost then density; energy by high ME; minerals by high content
        if nkey == "CP":
            order = sorted(candidate_ids, key=lambda i: (id2row[i]["cost"], id2row[i]["CP"]), reverse=True)
        elif nkey == "ME":
            order = sorted(candidate_ids, key=lambda i: id2row[i]["ME"], reverse=True)
        else:
            order = sorted(candidate_ids, key=lambda i: id2row[i][nkey], reverse=True)

        fid = pick_filler_for(nkey)
        if fid is None:
            return 0.0

        trimmed = 0.0
        remaining = surplus0
        for iid in order:
            if remaining <= 1e-9:
                break
            want = headroom_down(iid)
            if want <= 1e-9:
                continue
            cut = safe_cut_with_fill(iid, fid, nkey, want)
            if cut <= 1e-9:
                continue
            sol[iid] -= cut
            sol[fid] = sol.get(fid, 0.0) + cut
            trimmed += cut

            pnow = profile(sol)
            remaining = pnow[nkey] - targets[f"{nkey}_min"]

        return trimmed

    for _ in range(5):
        p = profile(sol)

        # 1) CP surplus -> proteins
        if p["CP"] - targets["CP_min"] > 1e-9 and protein_ids:
            trim_group("CP", protein_ids)

        # 2) ME surplus -> oil first, then other energy
        p = profile(sol)
        if p["ME"] - targets["ME_min"] > 1e-9 and energy_ids:
            if oil_id:
                trim_group("ME", [oil_id])
            p = profile(sol)
            if p["ME"] - targets["ME_min"] > 1e-9 and non_oil_energy:
                trim_group("ME", non_oil_energy)

        # 3) Minerals with dual logic
        p = profile(sol)
        ca_sur = p["Ca"] - targets["Ca_min"]
        p_sur = p["P"] - targets["P_min"]

        mineral_duals = [i for i in mineral_ids if id2row[i]["Ca"] > 0 and id2row[i]["P"] > 0]  # e.g., DCP
        mineral_ca_only = [i for i in mineral_ids if id2row[i]["Ca"] > 0 and id2row[i]["P"] == 0]
        mineral_p_only = [i for i in mineral_ids if id2row[i]["P"] > 0 and id2row[i]["Ca"] == 0]

        # If both Ca & P are high, trim duals first
        if ca_sur > 1e-9 and p_sur > 1e-9 and mineral_duals:
            trim_group("Ca", mineral_duals)
            trim_group("P", mineral_duals)

        # Now single-side surpluses
        p = profile(sol)
        if p["Ca"] - targets["Ca_min"] > 1e-9 and mineral_ca_only:
            trim_group("Ca", mineral_ca_only)
        p = profile(sol)
        if p["P"] - targets["P_min"] > 1e-9 and mineral_p_only:
            trim_group("P", mineral_p_only)

        # If any other mineral overshoots exist (Lys/Met/NaCl), trim their strongest sources
        for k in ["Lys", "Met", "NaCl"]:
            p = profile(sol)
            if p[k] - targets[f"{k}_min"] > 1e-9:
                cands = [i for i in mineral_ids if id2row[i][k] > 0]
                if cands:
                    trim_group(k, cands)

        # Keep minerals under reserved cap (no top-up; just enforce cap if moved)
        used_minerals = sum(sol.get(i, 0.0) for i in mineral_ids)
        if used_minerals > minerals_reserved_cap + 1e-9:
            scale = minerals_reserved_cap / used_minerals if used_minerals > 0 else 1.0
            delta = 0.0
            for i in mineral_ids:
                old = sol.get(i, 0.0)
                new = old * scale
                sol[i] = new
                delta += (old - new)
            # move freed % to cheapest non-mineral with headroom (keeps 100%)
            if delta > 1e-9:
                pool = [i for i in sol if i not in mineral_ids and headroom_up(i) > 0]
                pool.sort(key=lambda j: id2row[j]["cost"])
                for j in pool:
                    give = min(headroom_up(j), delta)
                    if give > 1e-9:
                        sol[j] += give
                        delta -= give
                    if delta <= 1e-9:
                        break

    # ---------------------------
    # Final rebalancing:
    # 1) energy surplus redistribute
    # 2) CP-only trim (if CP > CP_min + 0.02)
    # ---------------------------
    def redistribute_energy_surplus(max_rounds: int = 10):
        """
        Last defence against 'too-dense energy' problem.
        When ME target is low (e.g., layers), LP may pack high-ME ingredients.
        This step shifts % from high-ME expensive donors → low-ME cheap fillers
        while enforcing ALL mins and CF_max. Total always = 100%.
        """
        for _round in range(max_rounds):
            p_now = profile(sol)
            surplus_me = p_now["ME"] - targets["ME_min"]
            if surplus_me <= 0.5:  # kcal/kg tolerance
                break

            # donors: high-ME energy sources with some space to cut
            donors = [
                i for i in non_oil_energy
                if sol.get(i, 0.0) > 0.01 and id2row[i]["ME"] > 2500
            ]
            if not donors:
                break
            donors.sort(key=lambda i: (id2row[i]["ME"], id2row[i]["cost"]), reverse=True)

            # fillers: non-mineral ingredients with headroom and some ME
            fillers = [
                i for i in id2row
                if i not in mineral_ids
                and headroom_up(i) > 0.01
                and id2row[i]["ME"] > 0
            ]
            if not fillers:
                break
            fillers.sort(key=lambda i: (id2row[i]["ME"], id2row[i]["cost"]))  # low ME + cheap

            moved_any = False

            for d in donors:
                for f in fillers:
                    if d == f:
                        continue
                    dens_d = id2row[d]["ME"] / 100.0
                    dens_f = id2row[f]["ME"] / 100.0
                    if dens_f >= dens_d:
                        continue

                    max_possible = min(headroom_down(d), headroom_up(f))
                    if max_possible <= 0.01:
                        continue

                    delta_per = dens_f - dens_d  # negative
                    if delta_per < 0:
                        max_me = surplus_me / (-delta_per)
                        max_possible = min(max_possible, max_me)

                    if max_possible <= 0.01:
                        continue

                    cut = safe_cut_with_fill(d, f, "ME", max_possible)
                    if cut <= 0.01:
                        continue

                    sol[d] -= cut
                    sol[f] = sol.get(f, 0.0) + cut
                    moved_any = True
                    break

                if moved_any:
                    break

            if not moved_any:
                break

    # filler for final protein trim: global preference
    def pick_filler_global(exclude: set[Any] | None = None) -> Any | None:
        """
        Filler preference:
        1) Cheap non-oil energy sources with headroom
        2) Medium sources
        3) Protein sources
        4) Finally oil (if really needed)
        """
        if exclude is None:
            exclude = set()

        candidates: List[Any] = []

        # 1) energy (non-oil)
        for i in non_oil_energy:
            if i in exclude:
                continue
            if headroom_up(i) > 1e-3:
                candidates.append(i)

        # 2) medium
        for i in medium_ids:
            if i in exclude:
                continue
            if headroom_up(i) > 1e-3:
                candidates.append(i)

        # 3) protein
        for i in protein_ids:
            if i in exclude:
                continue
            if headroom_up(i) > 1e-3:
                candidates.append(i)

        # 4) oil (last resort)
        if oil_id and oil_id not in exclude and headroom_up(oil_id) > 1e-3:
            candidates.append(oil_id)

        if not candidates:
            return None

        candidates = list(dict.fromkeys(candidates))  # unique, preserve order
        candidates.sort(key=lambda j: id2row[j]["cost"])
        return candidates[0]

    def final_protein_trim(max_rounds: int = 10):
        """
        Last correction for protein:
        - Check CP surplus above CP_min.
        - If surplus > 0.02 (% CP), trim expensive, high-CP protein sources.
        - Move same % into cheap filler (energy -> medium -> protein -> oil).
        All constraints (other mins + CF) are protected by safe_cut_with_fill.
        """
        tol_cp = 0.002  # CP tolerance (% units)

        for _ in range(max_rounds):
            prof = profile(sol)
            surplus_cp = prof["CP"] - targets["CP_min"]
            if surplus_cp <= tol_cp:
                break  # CP is close enough to target

            # pick donors: protein ingredients with some room to reduce
            donors = [
                i for i in protein_ids
                if sol.get(i, 0.0) > 0.01 and headroom_down(i) > 1e-3
            ]
            if not donors:
                break

            # order donors: highest cost + highest CP first
            donors.sort(key=lambda i: (id2row[i]["cost"], id2row[i]["CP"]), reverse=True)

            changed_any = False
            for d in donors:
                filler = pick_filler_global(exclude={d})
                if filler is None:
                    continue

                want = headroom_down(d)
                if want <= 1e-3:
                    continue

                # Cut d and add same % to filler with CP as driver
                cut = safe_cut_with_fill(d, filler, "CP", want)
                if cut <= 1e-3:
                    continue

                sol[d] -= cut
                sol[filler] = sol.get(filler, 0.0) + cut
                changed_any = True
                break  # re-check CP with updated solution

            if not changed_any:
                break

    # apply final rebalancing
    redistribute_energy_surplus()
    final_protein_trim()

    # ---------------------------
    # Build response
    # ---------------------------
    def achieved(k: str) -> float:
        """
        Achieved nutrients for reporting.
        IMPORTANT: Here we use REAL ME including minerals,
        so you can SEE the energy that minerals contribute.
        """
        return sum((id2row[i][k] * sol.get(i, 0.0) / 100.0) for i in sol)

    # Cap achieved values at target values - don't show values exceeding target
    def capped_achieved(nutrient_key: str, target_key: str) -> float:
        """Return achieved value capped at target if it exceeds target."""
        achieved_val = achieved(nutrient_key)
        target_val = targets.get(target_key, achieved_val)
        # If achieved exceeds target, cap it at target; otherwise keep achieved
        return min(achieved_val, target_val) if achieved_val > target_val else achieved_val
    
    achieved_map = {
        "ME_kcal_per_kg": round(capped_achieved("ME", "ME_min"), 4),
        "CP_percent": round(capped_achieved("CP", "CP_min"), 4),
        "Ca_percent": round(capped_achieved("Ca", "Ca_min"), 4),
        "P_percent": round(capped_achieved("P", "P_min"), 4),
        "Lys_percent": round(capped_achieved("Lys", "Lys_min"), 4),
        "Met_percent": round(capped_achieved("Met", "Met_min"), 4),
        "NaCl_percent": round(capped_achieved("NaCl", "NaCl_min"), 4),
        "CF_percent": round(achieved("CF"), 4),  # CF doesn't have a target to cap against
    }

    def name_of(iid):
        if isinstance(iid, str) and iid.startswith("premix:"):
            return next((r["name"] for r in rows if r["id"] == iid), "Premix")
        return id2row[iid]["name"]

    def cat_of(iid):
        return id2row[iid]["category"]

    # Do NOT include premix rows in final rows (UI shows premix separately)
    final_rows = [
        {"id": iid, "name": name_of(iid), "category": cat_of(iid), "percent": round(pct, 4)}
        for iid, pct in sorted(sol.items(), key=lambda kv: (-kv[1], name_of(kv[0])))
        if pct > 1e-6 and not (isinstance(iid, str) and iid.startswith("premix:"))
    ]
    total_pct = round(sum(sol.values()), 4)

    targets_out = {
        "ME_min": targets["ME_min"],
        "CP_min": targets["CP_min"],
        "Ca_min": targets["Ca_min"],
        "P_min": targets["P_min"],
        "Lys_min": targets["Lys_min"],
        "Met_min": targets["Met_min"],
        "NaCl_min": targets["NaCl_min"],
    }

    notes: List[str] = []
    if oil_added:
        notes.append(f"Oil present after trim: {sol.get(oil_id, 0.0):.3f}%." if oil_id else "No oil added.")
    if minerals_reserved_cap > 0:
        used_mins = sum(sol.get(i, 0.0) for i in mineral_ids)
        notes.append(f"Minerals used {used_mins:.3f}% of reserved {minerals_reserved_cap:.3f}% (post-trim).")

    # --- Per-ingredient nutrient contributions (all ingredients) ---
    ingredient_contrib_rows = []
    for iid, pct in sol.items():
        row = id2row[iid]
        if pct <= 1e-6:
            continue
        ingredient_contrib_rows.append({
            "name": row["name"],
            "category": row["category"],
            "percent": round(pct, 4),
            "ME": round(row["ME"] * pct / 100.0, 4),
            "CP": round(row["CP"] * pct / 100.0, 4),
            "Ca": round(row["Ca"] * pct / 100.0, 4),
            "P": round(row["P"] * pct / 100.0, 4),
            "Lys": round(row["Lys"] * pct / 100.0, 4),
            "Met": round(row["Met"] * pct / 100.0, 4),
            "NaCl": round(row["NaCl"] * pct / 100.0, 4),
            "CF": round(row["CF"] * pct / 100.0, 4),
        })

    # Mineral-only slice for the quick mineral table
    mineral_contrib_rows = [
        {
            "name": r["name"],
            "percent": r["percent"],
            "Ca": r["Ca"], "P": r["P"],
            "Lys": r["Lys"], "Met": r["Met"],
            "NaCl": r["NaCl"],
        }
        for r in ingredient_contrib_rows if r["category"] == "mineral"
    ]

    return JsonResponse({
        "status": "ok",
        "notes": notes,
        "premix": (
            {
                "id": premix_row["id"],
                "name": premix_row["name"],
                "inclusion_rate_percent": premix_row["min"],
            } if premix_row else None
        ),
        "final_formula": {
            "rows": final_rows,
            "totals": {"percent_sum": total_pct},
            "nutrients_achieved": {
                "ME_kcal_per_kg": achieved_map["ME_kcal_per_kg"],
                "CP_percent": achieved_map["CP_percent"],
                "Ca_percent": achieved_map["Ca_percent"],
                "P_percent": achieved_map["P_percent"],
                "Lys_percent": achieved_map["Lys_percent"],
                "Met_percent": achieved_map["Met_percent"],
                "NaCl_percent": achieved_map["NaCl_percent"],
            },
            "targets": targets_out,
        },
        "taxonomy": {
            "species": {"id": species.id, "name": species.name},
            "subspecies": {"id": subspecies.id, "name": subspecies.name},
            "animal_type": {"id": animal_type.id, "name": animal_type.name},
            "phase": {"id": phase.id, "name": phase.name},
        },
        "ingredient_contributions": ingredient_contrib_rows,
        "mineral_contributions": mineral_contrib_rows,
    }, status=200)


@login_required
@require_http_methods(["GET"])
def contributions(request):
    """Display mineral and ingredient contributions page"""
    return render(request, 'core/contributions.html')


@login_required
@require_http_methods(["GET"])
def export_contributions_pdf(request):
    """Export contributions as PDF"""
    if not REPORTLAB_AVAILABLE:
        return HttpResponse("PDF export requires reportlab library. Please install it: pip install reportlab==4.0.7", status=500)
    try:
        data_str = request.GET.get('data', '{}')
        data = json.loads(data_str) if data_str else {}
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), topMargin=0.5*inch)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=20,
            alignment=1  # Center
        )
        story.append(Paragraph("Feed Formulation - Contributions Report", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Taxonomy info
        taxonomy = data.get('taxonomy', {})
        if taxonomy:
            info_text = f"Species: {taxonomy.get('species', {}).get('name', 'N/A')} | "
            info_text += f"Subspecies: {taxonomy.get('subspecies', {}).get('name', 'N/A')} | "
            info_text += f"Animal Type: {taxonomy.get('animal_type', {}).get('name', 'N/A')} | "
            info_text += f"Phase: {taxonomy.get('phase', {}).get('name', 'N/A')}"
            story.append(Paragraph(info_text, styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Mineral Contributions Table
        minerals = data.get('mineral_contributions', [])
        if minerals:
            story.append(Paragraph("<b>Mineral Contributions</b>", styles['Heading2']))
            story.append(Spacer(1, 0.1*inch))
            
            mineral_data = [['Mineral', '%', 'Ca (%)', 'P (%)', 'Lys (%)', 'Met (%)', 'Salt (%)']]
            for m in minerals:
                mineral_data.append([
                    m.get('name', ''),
                    f"{m.get('percent', 0):.4f}",
                    f"{m.get('Ca', 0):.4f}",
                    f"{m.get('P', 0):.4f}",
                    f"{m.get('Lys', 0):.4f}",
                    f"{m.get('Met', 0):.4f}",
                    f"{m.get('NaCl', 0):.4f}",
                ])
            
            mineral_table = Table(mineral_data)
            mineral_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(mineral_table)
            story.append(Spacer(1, 0.3*inch))
        
        # All Ingredient Contributions Table
        ingredients = data.get('ingredient_contributions', [])
        if ingredients:
            story.append(Paragraph("<b>All Ingredient Contributions</b>", styles['Heading2']))
            story.append(Spacer(1, 0.1*inch))
            
            ing_data = [['Ingredient', 'Category', 'Inclusion %', 'ME (kcal/kg)', 'CP (%)', 'Ca (%)', 'P (%)', 'Lys (%)', 'Met (%)', 'Salt (%)', 'CF (%)']]
            for ing in ingredients:
                ing_data.append([
                    ing.get('name', ''),
                    ing.get('category', ''),
                    f"{ing.get('percent', 0):.4f}",
                    f"{ing.get('ME', 0):.2f}",
                    f"{ing.get('CP', 0):.4f}",
                    f"{ing.get('Ca', 0):.4f}",
                    f"{ing.get('P', 0):.4f}",
                    f"{ing.get('Lys', 0):.4f}",
                    f"{ing.get('Met', 0):.4f}",
                    f"{ing.get('NaCl', 0):.4f}",
                    f"{ing.get('CF', 0):.4f}",
                ])
            
            ing_table = Table(ing_data, colWidths=[1.2*inch]*11)
            ing_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 8),
                ('FONTSIZE', (0, 1), (-1, -1), 7),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(ing_table)
        
        doc.build(story)
        buffer.seek(0)
        
        response = HttpResponse(buffer.read(), content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="feed_contributions.pdf"'
        return response
        
    except Exception as e:
        return HttpResponse(f"Error generating PDF: {str(e)}", status=500)


@login_required
@require_http_methods(["GET"])
def export_contributions_excel(request):
    """Export contributions as Excel"""
    if not OPENPYXL_AVAILABLE:
        return HttpResponse("Excel export requires openpyxl library. Please install it: pip install openpyxl==3.1.2", status=500)
    try:
        data_str = request.GET.get('data', '{}')
        data = json.loads(data_str) if data_str else {}
        
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Contributions"
        
        # Header style
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF", size=11)
        border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        row = 1
        
        # Title
        ws.merge_cells(f'A{row}:K{row}')
        title_cell = ws[f'A{row}']
        title_cell.value = "Feed Formulation - Contributions Report"
        title_cell.font = Font(bold=True, size=14)
        title_cell.alignment = Alignment(horizontal='center', vertical='center')
        row += 2
        
        # Taxonomy info
        taxonomy = data.get('taxonomy', {})
        if taxonomy:
            ws.merge_cells(f'A{row}:K{row}')
            info_cell = ws[f'A{row}']
            info_text = f"Species: {taxonomy.get('species', {}).get('name', 'N/A')} | "
            info_text += f"Subspecies: {taxonomy.get('subspecies', {}).get('name', 'N/A')} | "
            info_text += f"Animal Type: {taxonomy.get('animal_type', {}).get('name', 'N/A')} | "
            info_text += f"Phase: {taxonomy.get('phase', {}).get('name', 'N/A')}"
            info_cell.value = info_text
            info_cell.font = Font(size=10)
            row += 2
        
        # Mineral Contributions
        minerals = data.get('mineral_contributions', [])
        if minerals:
            ws[f'A{row}'] = "Mineral Contributions"
            ws[f'A{row}'].font = Font(bold=True, size=12)
            row += 1
            
            # Headers
            headers = ['Mineral', '%', 'Ca (%)', 'P (%)', 'Lys (%)', 'Met (%)', 'Salt (%)']
            for col, header in enumerate(headers, 1):
                cell = ws.cell(row=row, column=col)
                cell.value = header
                cell.fill = header_fill
                cell.font = header_font
                cell.border = border
                cell.alignment = Alignment(horizontal='center', vertical='center')
            row += 1
            
            # Data
            for m in minerals:
                ws.cell(row=row, column=1, value=m.get('name', '')).border = border
                ws.cell(row=row, column=2, value=round(m.get('percent', 0), 4)).border = border
                ws.cell(row=row, column=3, value=round(m.get('Ca', 0), 4)).border = border
                ws.cell(row=row, column=4, value=round(m.get('P', 0), 4)).border = border
                ws.cell(row=row, column=5, value=round(m.get('Lys', 0), 4)).border = border
                ws.cell(row=row, column=6, value=round(m.get('Met', 0), 4)).border = border
                ws.cell(row=row, column=7, value=round(m.get('NaCl', 0), 4)).border = border
                for col in range(1, 8):
                    ws.cell(row=row, column=col).alignment = Alignment(horizontal='center')
                row += 1
            
            row += 1
        
        # All Ingredient Contributions
        ingredients = data.get('ingredient_contributions', [])
        if ingredients:
            ws[f'A{row}'] = "All Ingredient Contributions"
            ws[f'A{row}'].font = Font(bold=True, size=12)
            row += 1
            
            # Headers
            headers = ['Ingredient', 'Category', 'Inclusion %', 'ME (kcal/kg)', 'CP (%)', 'Ca (%)', 'P (%)', 'Lys (%)', 'Met (%)', 'Salt (%)', 'CF (%)']
            for col, header in enumerate(headers, 1):
                cell = ws.cell(row=row, column=col)
                cell.value = header
                cell.fill = header_fill
                cell.font = header_font
                cell.border = border
                cell.alignment = Alignment(horizontal='center', vertical='center')
            row += 1
            
            # Data
            for ing in ingredients:
                ws.cell(row=row, column=1, value=ing.get('name', '')).border = border
                ws.cell(row=row, column=2, value=ing.get('category', '')).border = border
                ws.cell(row=row, column=3, value=round(ing.get('percent', 0), 4)).border = border
                ws.cell(row=row, column=4, value=round(ing.get('ME', 0), 2)).border = border
                ws.cell(row=row, column=5, value=round(ing.get('CP', 0), 4)).border = border
                ws.cell(row=row, column=6, value=round(ing.get('Ca', 0), 4)).border = border
                ws.cell(row=row, column=7, value=round(ing.get('P', 0), 4)).border = border
                ws.cell(row=row, column=8, value=round(ing.get('Lys', 0), 4)).border = border
                ws.cell(row=row, column=9, value=round(ing.get('Met', 0), 4)).border = border
                ws.cell(row=row, column=10, value=round(ing.get('NaCl', 0), 4)).border = border
                ws.cell(row=row, column=11, value=round(ing.get('CF', 0), 4)).border = border
                for col in range(1, 12):
                    ws.cell(row=row, column=col).alignment = Alignment(horizontal='center')
                row += 1
        
        # Auto-adjust column widths
        from openpyxl.utils import get_column_letter
        from openpyxl.cell.cell import MergedCell
        
        for col_idx, column in enumerate(ws.columns, 1):
            max_length = 0
            # Skip if first cell is a MergedCell (can't get column_letter from it)
            first_cell = column[0]
            if isinstance(first_cell, MergedCell):
                # Get column letter from index instead
                column_letter = get_column_letter(col_idx)
            else:
                column_letter = first_cell.column_letter
            
            for cell in column:
                # Skip MergedCell objects
                if isinstance(cell, MergedCell):
                    continue
                try:
                    if cell.value and len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            if max_length > 0:
                adjusted_width = min(max_length + 2, 30)
                ws.column_dimensions[column_letter].width = adjusted_width
        
        buffer = io.BytesIO()
        wb.save(buffer)
        buffer.seek(0)
        
        response = HttpResponse(buffer.read(), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response['Content-Disposition'] = 'attachment; filename="feed_contributions.xlsx"'
        return response
        
    except Exception as e:
        return HttpResponse(f"Error generating Excel: {str(e)}", status=500)

