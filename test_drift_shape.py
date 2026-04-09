import unittest
import numpy as np
import json
from pathlib import Path
import importlib.util

# Load drift module by path
SPEC_PATH = Path(__file__).parent / "drift.py"
spec = importlib.util.spec_from_file_location("v2_drift", str(SPEC_PATH))
drift = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drift)

class TestDriftShapes(unittest.TestCase):
    """Test mathematical correctness of drift term shapes and scaling."""
    
    def setUp(self):
        """Initialize synthetic data for testing."""
        # Synthetic coordinates
        self.x = np.linspace(0, 100, 20)
        self.y = np.linspace(0, 50, 20)
        
        # Compute resc using standard parameters
        covmax = 1.0
        variogram_range = 100.0
        self.resc = drift.compute_resc(covmax, self.x, self.y, variogram_range)
        
        # Verify resc is positive and reasonable
        self.assertGreater(self.resc, 0, "resc must be positive")
        self.assertLess(self.resc, 1.0, "resc should be < 1.0 after inversion")

    def test_linear_x_shape_and_scaling(self):
        """Test linear_x term: D = resc * x."""
        config = {"drift_terms": {"linear_x": True}}
        drift_matrix, term_names = drift.compute_polynomial_drift(
            self.x, self.y, config, self.resc
        )
        
        results = drift.verify_drift_physics(
            drift_matrix, term_names, self.x, self.y, self.resc
        )
        
        self.assertIn("linear_x", results)
        self.assertEqual(results["linear_x"], "PASS",
                        f"linear_x verification failed: {results}")
    
    def test_linear_y_shape_and_scaling(self):
        """Test linear_y term: D = resc * y."""
        config = {"drift_terms": {"linear_y": True}}
        drift_matrix, term_names = drift.compute_polynomial_drift(
            self.x, self.y, config, self.resc
        )
        
        results = drift.verify_drift_physics(
            drift_matrix, term_names, self.x, self.y, self.resc
        )
        
        self.assertIn("linear_y", results)
        self.assertEqual(results["linear_y"], "PASS",
                        f"linear_y verification failed: {results}")
    
    def test_quadratic_x_shape_and_scaling(self):
        """Test quadratic_x term: D = resc * x²."""
        config = {"drift_terms": {"quadratic_x": True}}
        drift_matrix, term_names = drift.compute_polynomial_drift(
            self.x, self.y, config, self.resc
        )
        
        results = drift.verify_drift_physics(
            drift_matrix, term_names, self.x, self.y, self.resc
        )
        
        self.assertIn("quadratic_x", results)
        self.assertEqual(results["quadratic_x"], "PASS",
                        f"quadratic_x verification failed: {results}")
    
    def test_quadratic_y_shape_and_scaling(self):
        """Test quadratic_y term: D = resc * y²."""
        config = {"drift_terms": {"quadratic_y": True}}
        drift_matrix, term_names = drift.compute_polynomial_drift(
            self.x, self.y, config, self.resc
        )
        
        results = drift.verify_drift_physics(
            drift_matrix, term_names, self.x, self.y, self.resc
        )
        
        self.assertIn("quadratic_y", results)
        self.assertEqual(results["quadratic_y"], "PASS",
                        f"quadratic_y verification failed: {results}")

    def test_all_terms_together(self):
        """Test all 4 terms enabled simultaneously."""
        config = {
            "drift_terms": {
                "linear_x": True,
                "linear_y": True,
                "quadratic_x": True,
                "quadratic_y": True
            }
        }
        drift_matrix, term_names = drift.compute_polynomial_drift(
            self.x, self.y, config, self.resc
        )
        
        results = drift.verify_drift_physics(
            drift_matrix, term_names, self.x, self.y, self.resc
        )
        
        # All terms should pass
        expected_terms = ["linear_x", "linear_y", "quadratic_x", "quadratic_y"]
        for term in expected_terms:
            self.assertIn(term, results, f"Missing term: {term}")
            self.assertEqual(results[term], "PASS",
                           f"{term} verification failed: {results[term]}")

    def test_empty_drift_matrix(self):
        """Test with no drift terms enabled."""
        config = {"drift_terms": {}}
        drift_matrix, term_names = drift.compute_polynomial_drift(
            self.x, self.y, config, self.resc
        )
        
        results = drift.verify_drift_physics(
            drift_matrix, term_names, self.x, self.y, self.resc
        )
        
        self.assertEqual(results, {}, "Empty drift should return empty results")
    
    def test_collinear_data(self):
        """Test with collinear data (y_range = 0)."""
        x_col = np.linspace(0, 100, 20)
        y_col = np.zeros(20)  # All same y value
        
        config = {"drift_terms": {"linear_x": True}}
        drift_matrix, term_names = drift.compute_polynomial_drift(
            x_col, y_col, config, self.resc
        )
        
        results = drift.verify_drift_physics(
            drift_matrix, term_names, x_col, y_col, self.resc
        )
        
        # Should still pass because linear_x depends only on x
        self.assertEqual(results["linear_x"], "PASS")
    
    def test_single_point(self):
        """Test with single data point."""
        x_single = np.array([50.0])
        y_single = np.array([25.0])
        
        config = {"drift_terms": {"linear_x": True, "linear_y": True}}
        drift_matrix, term_names = drift.compute_polynomial_drift(
            x_single, y_single, config, self.resc
        )
        
        # With single point, R² is undefined (0 variance)
        # Function should handle gracefully
        results = drift.verify_drift_physics(
            drift_matrix, term_names, x_single, y_single, self.resc
        )
        
        # May FAIL due to R² = NaN, but should not crash
        self.assertIsNotNone(results)

    def test_deterministic_ordering(self):
        """Verify term order is always fixed regardless of config dict order."""
        # Create config with terms in random order
        config = {
            "drift_terms": {
                "quadratic_y": True,
                "linear_x": True,
                "quadratic_x": True,
                "linear_y": True
            }
        }
        
        drift_matrix, term_names = drift.compute_polynomial_drift(
            self.x, self.y, config, self.resc
        )
        
        # Order should always be: linear_x, linear_y, quadratic_x, quadratic_y
        expected_order = ["linear_x", "linear_y", "quadratic_x", "quadratic_y"]
        self.assertEqual(term_names, expected_order,
                        f"Term order not deterministic. Got {term_names}")

if __name__ == "__main__":
    unittest.main(verbosity=2)
