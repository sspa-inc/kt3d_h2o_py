# Variogram Models Reference

**Source:** [`variogram.py:66-81`](../variogram.py#L66)

---

## Overview

A variogram model describes how spatial correlation decays with distance. The semivariance `γ(h)` is a function of the lag distance `h` between two points. As `h` increases, `γ(h)` rises from the nugget toward the sill.

### Parameter Definitions

| Parameter | Symbol | Description |
|---|---|---|
| `sill` | C | Total variance; the plateau value that `γ(h)` approaches at large `h` |
| `nugget` | C₀ | Discontinuity at the origin; represents micro-scale variability or measurement error |
| `range` | a | Distance at which the variogram reaches (or effectively reaches) the sill |
| `psill` | C − C₀ | Partial sill; the variance attributable to spatial structure (`psill = sill - nugget`) |

> **Note:** The `sill` parameter in config is the **total** sill (nugget + psill). The `psill` is computed internally as `sill - nugget`.

---

## Supported Models

### 1. Linear

```
γ(h) = nugget + (psill / range) * h    for h ≤ range
γ(h) = sill                            for h > range
```

The linear model increases linearly with distance up to the range, then plateaus at the sill. It is bounded (reaches the sill exactly at `h = range`).

**Implementation:** [`variogram.py:69-70`](../variogram.py#L69)

```python
nugget + (psill / range_) * h_val if h_val <= range_ else sill
```

**Characteristics:**
- Bounded model (reaches sill at `h = range`)
- No differentiability at `h = range`
- Suitable when spatial correlation drops off linearly

---

### 2. Exponential

```
γ(h) = nugget + psill * (1 - exp(-h / (range/3)))
```

The exponential model approaches the sill asymptotically. The `range/3` factor is applied so that the model reaches approximately 95% of the sill at `h = range` (the "practical range").

**Implementation:** [`variogram.py:72`](../variogram.py#L72)

```python
nugget + psill * (1 - math.exp(-h_val / (range_ / 3.0)))
```

**Characteristics:**
- Unbounded (asymptotically approaches sill)
- Reaches ~95% of sill at `h = range` (practical range convention)
- Smooth at origin; suitable for gradual spatial transitions

---

### 3. Spherical

```
γ(h) = nugget + psill * (1.5*(h/range) - 0.5*(h/range)³)    for h ≤ range
γ(h) = sill                                                   for h > range
```

The spherical model is the most commonly used in geostatistics. It rises steeply near the origin and reaches the sill exactly at `h = range`.

**Implementation:** [`variogram.py:73-77`](../variogram.py#L73)

```python
if h_val <= range_:
    nugget + psill * (1.5 * (h_val / range_) - 0.5 * (h_val / range_)**3)
else:
    sill
```

**Characteristics:**
- Bounded model (reaches sill exactly at `h = range`)
- Linear behavior near origin
- Most widely used model in hydrogeology

---

### 4. Gaussian

```
γ(h) = nugget + psill * (1 - exp(-(h / (range/√3))²))
```

The Gaussian model has a parabolic behavior near the origin, indicating very smooth spatial variation. The `range/√3` factor ensures the model reaches approximately 95% of the sill at `h = range`.

**Implementation:** [`variogram.py:79`](../variogram.py#L79)

```python
nugget + psill * (1 - math.exp(-(h_val / (range_ / math.sqrt(3.0)))**2))
```

**Characteristics:**
- Unbounded (asymptotically approaches sill)
- Parabolic near origin; implies very smooth spatial fields
- Reaches ~95% of sill at `h = range` (practical range convention)
- Can cause numerical instability for large datasets; use with care

---

## Model Comparison Plot

The following plot shows all four models with identical parameters: `sill=1`, `range=100`, `nugget=0.1`.

```
γ(h)
1.0 |─────────────────────────────────────────────────── sill
    |                          ╭──── Spherical (reaches sill at h=range)
    |                    ╭─────╯     Linear (reaches sill at h=range)
0.5 |              ╭─────╯           Exponential (asymptotic)
    |         ╭────╯                 Gaussian (parabolic near origin)
    |    ╭────╯
0.1 |────╯  ← nugget
    |
    +────────────────────────────────────────────────────→ h
    0       25      50      75     100     125     150
                                   ↑
                                 range
```

To reproduce this plot programmatically:

```python
import numpy as np
import matplotlib.pyplot as plt
import math

sill, range_, nugget = 1.0, 100.0, 0.1
psill = sill - nugget
h = np.linspace(0, 200, 500)

def linear(h):
    return np.where(h <= range_, nugget + (psill / range_) * h, sill)

def exponential(h):
    return nugget + psill * (1 - np.exp(-h / (range_ / 3.0)))

def spherical(h):
    return np.where(
        h <= range_,
        nugget + psill * (1.5 * (h / range_) - 0.5 * (h / range_)**3),
        sill
    )

def gaussian(h):
    return nugget + psill * (1 - np.exp(-(h / (range_ / math.sqrt(3.0)))**2))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(h, linear(h),      label="Linear")
ax.plot(h, exponential(h), label="Exponential")
ax.plot(h, spherical(h),   label="Spherical")
ax.plot(h, gaussian(h),    label="Gaussian")
ax.axhline(sill,   color='gray', linestyle='--', label=f"Sill = {sill}")
ax.axhline(nugget, color='gray', linestyle=':',  label=f"Nugget = {nugget}")
ax.axvline(range_, color='gray', linestyle='-.',  label=f"Range = {range_}")
ax.set_xlabel("Lag distance h")
ax.set_ylabel("Semivariance γ(h)")
ax.set_title("Variogram Models (sill=1, range=100, nugget=0.1)")
ax.legend()
plt.tight_layout()
plt.show()
```

---

## The `effective_range_convention` Parameter

**Config path:** `variogram.advanced.effective_range_convention`  
**Type:** `bool`  
**Default:** `true`

This parameter controls how the `range` value is interpreted for **asymptotic models** (exponential and Gaussian).

| Value | Meaning |
|---|---|
| `true` (default) | `range` is the **practical range** — the distance at which the model reaches ~95% of the sill. The internal scale parameter is adjusted accordingly (`range/3` for exponential, `range/√3` for Gaussian). |
| `false` | `range` is the **model parameter** passed directly to the formula without adjustment. |

**Effect on formulas:**

When `effective_range_convention = true` (default):
- Exponential: `γ(h) = nugget + psill * (1 - exp(-h / (range/3)))`
- Gaussian: `γ(h) = nugget + psill * (1 - exp(-(h / (range/√3))²))`

When `effective_range_convention = false`:
- Exponential: `γ(h) = nugget + psill * (1 - exp(-h / range))`
- Gaussian: `γ(h) = nugget + psill * (1 - exp(-(h / range)²))`

> **Note:** For bounded models (spherical, linear), `effective_range_convention` has no effect — the range is always the exact distance at which the sill is reached.

> **Practical guidance:** Leave `effective_range_convention = true` (the default) unless you are fitting variogram parameters from a tool that uses the raw model parameter convention. With the default, the `range` value you specify in config directly corresponds to the correlation length visible in an experimental variogram.

---

## Validation Rules

The [`variogram`](../variogram.py#L5) class enforces the following at construction time (see [`_validate_basic_parameters()`](../variogram.py#L38)):

| Rule | Error Raised |
|---|---|
| `sill > 0` | `ValueError: Variogram sill must be positive` |
| `range > 0` | `ValueError: Variogram range must be positive` |
| `nugget >= 0` | `ValueError: Variogram nugget must be non-negative` |
| `nugget < sill` | `ValueError: Variogram nugget must be less than total sill` |

---

## Quick Reference Table

| Model | Bounded? | Behavior at Origin | Reaches Sill At | Typical Use |
|---|---|---|---|---|
| Linear | Yes | Linear | `h = range` (exact) | Simple, monotone correlation decay |
| Exponential | No (asymptotic) | Linear | `h ≈ range` (95%) | Gradual, smooth transitions |
| Spherical | Yes | Linear | `h = range` (exact) | Most common; hydrogeology standard |
| Gaussian | No (asymptotic) | Parabolic | `h ≈ range` (95%) | Very smooth fields; use cautiously |

---

## See Also

- [`docs/glossary.md`](../glossary.md) — definitions of sill, nugget, range, psill
- [`docs/api/variogram.md`](../api/variogram.md) — full API reference for the `variogram` class
- [`docs/validation/vv_variogram_models.py`](../validation/vv_variogram_models.py) — V&V script verifying model equations against analytical values
