import math
import json
import numpy as np

class variogram:
    def __init__(self, config=None, config_path="config.json"):
        if config is None:
            with open(config_path, 'r') as f:
                config = json.load(f)

        # Basic variogram parameters
        self.model_type = config["variogram"].get("model", "spherical")
        self.sill = float(config["variogram"].get("sill", 1.0))
        self.range_ = float(config["variogram"].get("range", 1000.0))
        self.nugget = float(config["variogram"].get("nugget", 0.0))

        # Anisotropy parameters
        anisotropy_config = config["variogram"].get("anisotropy", {})
        self.anisotropy_enabled = anisotropy_config.get("enabled", False)
        self.anisotropy_ratio = float(anisotropy_config.get("ratio", 1.0))
        # Azimuth convention: Clockwise from North, 0° = North (KT3D convention)
        self.angle_major = float(anisotropy_config.get("angle_major", 0.0))
        
        # REMOVED: angle_minor (Standard 2D anisotropy implies orthogonality)

        # Advanced parameters
        advanced_config = config["variogram"].get("advanced", {})
        self.effective_range_convention = advanced_config.get("effective_range_convention", True)
        self.search_radius = advanced_config.get("search_radius", None)
        self.max_neighbors = advanced_config.get("max_neighbors", None)
        self.min_neighbors = advanced_config.get("min_neighbors", None)

        self._validate_basic_parameters()
        if self.anisotropy_enabled:
            self._validate_anisotropy()
        self._validate_advanced_parameters()

    def _validate_basic_parameters(self):
        if self.sill <= 0:
            raise ValueError(f"Variogram sill must be positive, got {self.sill}")
        if self.range_ <= 0:
            raise ValueError(f"Variogram range must be positive, got {self.range_}")
        if self.nugget < 0:
            raise ValueError(f"Variogram nugget must be non-negative, got {self.nugget}")
        if self.nugget >= self.sill:
            raise ValueError(f"Variogram nugget ({self.nugget}) must be less than total sill ({self.sill})")

    def _validate_anisotropy(self):
        if not (0 < self.anisotropy_ratio <= 1):
            raise ValueError(f"Anisotropy ratio must be in (0, 1], got {self.anisotropy_ratio}")
        if not (0 <= self.angle_major < 360):
            raise ValueError(f"Angle major must be in [0, 360), got {self.angle_major}")

    def _validate_advanced_parameters(self):
        if self.search_radius is not None and self.search_radius <= 0:
            raise ValueError(f"Search radius must be positive or None, got {self.search_radius}")

    @property
    def model(self):
        return self.model_type

    @property
    def parameters(self):
        return {'sill': self.sill, 'range': self.range_, 'nugget': self.nugget}

    def _compute_h_isotropic(self, h_val):
        """Internal isotropic calculation logic."""
        psill = self.sill - self.nugget
        if self.model_type == "linear":
            return self.nugget + (psill / self.range_) * h_val if h_val <= self.range_ else self.sill
        elif self.model_type == "exponential":
            return self.nugget + psill * (1 - math.exp(-h_val / (self.range_ / 3.0)))
        elif self.model_type == "spherical":
            if h_val <= self.range_:
                return self.nugget + psill * (1.5 * (h_val / self.range_) - 0.5 * (h_val / self.range_)**3)
            else:
                return self.sill
        elif self.model_type == "gaussian":
            return self.nugget + psill * (1 - math.exp(-(h_val / (self.range_ / math.sqrt(3.0)))**2))
        else:
            raise ValueError("Invalid variogram model type.")

    def calculate_variogram(self, h):
        """Calculates semivariance for a scalar isotropic distance h."""
        if self.anisotropy_enabled:
            # If anisotropy is on, 'h' is ambiguous. Is it major axis distance? 
            # Or already transformed distance?
            # We assume the user has ALREADY transformed the coordinate if they are passing a scalar.
            pass 
        return self._compute_h_isotropic(h)

    def calculate_variogram_at_vector(self, hx, hy):
        """
        Calculates semivariance for a vector lag (hx, hy) accounting for anisotropy.
        Applies coordinate transformation locally to determine effective distance.
        """
        if not self.anisotropy_enabled:
            h_dist = math.sqrt(hx**2 + hy**2)
            return self._compute_h_isotropic(h_dist)

        # 1. Rotate vector so Major Axis aligns with X-axis (azimuth→arithmetic: alpha = 90 - azimuth)
        # We rotate by NEGATIVE angle to align the data's major axis to global X
        # Note: This logic must match transform.py's rotation matrix
        theta = math.radians(90.0 - self.angle_major)
        c, s = math.cos(theta), math.sin(theta)
        
        # Rotate coordinates
        # hx_rot = hx * cos(theta) + hy * sin(theta)
        # hy_rot = -hx * sin(theta) + hy * cos(theta)
        # This effectively projects h onto the major (X') and minor (Y') axes
        hx_prime = hx * c + hy * s
        hy_prime = -hx * s + hy * c
        
        # 2. Scale the Minor Axis (Y') to match Major Range
        # If ratio = 0.5 (Minor Range = 50, Major = 100).
        # We need to stretch Y by 2.0 so that distance 50 becomes 100.
        hy_scaled = hy_prime * (1.0 / self.anisotropy_ratio)
        
        # 3. Compute effective isotropic distance
        h_effective = math.sqrt(hx_prime**2 + hy_scaled**2)
        
        return self._compute_h_isotropic(h_effective)

    def clone(self):
        import copy
        return copy.deepcopy(self)