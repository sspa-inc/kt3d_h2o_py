import unittest
import numpy as np
import pandas as pd
import os
from shapely.geometry import LineString, MultiLineString
import geopandas as gpd

# Assuming data.py is in the same directory or accessible via PYTHONPATH
from v2_Code.data import load_line_features, load_observation_wells, prepare_data


class TestLoadLineFeatures(unittest.TestCase):

    def setUp(self):
        # Create dummy shapefiles for testing
        self.test_dir = "test_line_features"
        os.makedirs(self.test_dir, exist_ok=True)

        # Test case 1: Simple LineString with stage values
        self.line1 = LineString([(0, 0), (100, 100)])
        self.gdf1 = gpd.GeoDataFrame({
            'geometry': [self.line1],
            'stage_start': [10.0],
            'stage_end': [20.0]
        })
        self.path1 = os.path.join(self.test_dir, "line1.shp")
        self.gdf1.to_file(self.path1, driver='ESRI Shapefile')

        # Test case 2: MultiLineString with stage values
        self.line2a = LineString([(0, 0), (50, 50)])
        self.line2b = LineString([(50, 50), (100, 0)])
        self.gdf2 = gpd.GeoDataFrame({
            'geometry': [MultiLineString([self.line2a, self.line2b])],
            'stage_start': [15.0],
            'stage_end': [25.0]
        })
        self.path2 = os.path.join(self.test_dir, "line2.shp")
        self.gdf2.to_file(self.path2, driver='ESRI Shapefile')

        # Test case 3: LineString with missing stage columns
        self.line3 = LineString([(0, 0), (100, 0)])
        self.gdf3 = gpd.GeoDataFrame({
            'geometry': [self.line3],
            'other_col': [1.0]
        })
        self.path3 = os.path.join(self.test_dir, "line3.shp")
        self.gdf3.to_file(self.path3, driver='ESRI Shapefile')

        # Test case 4: Empty shapefile
        self.gdf4 = gpd.GeoDataFrame({
            'geometry': [],
            'stage_start': [],
            'stage_end': []
        })
        self.path4 = os.path.join(self.test_dir, "line4.shp")
        self.gdf4.to_file(self.path4, driver='ESRI Shapefile')

        # Test case 5: LineString with zero length
        self.line5 = LineString([(0, 0), (0, 0)])
        self.gdf5 = gpd.GeoDataFrame({
            'geometry': [self.line5],
            'stage_start': [10.0],
            'stage_end': [20.0]
        })
        self.path5 = os.path.join(self.test_dir, "line5.shp")
        self.gdf5.to_file(self.path5, driver='ESRI Shapefile')

        # Test case 6: LineString with spacing larger than length
        self.line6 = LineString([(0, 0), (50, 50)])
        self.gdf6 = gpd.GeoDataFrame({
            'geometry': [self.line6],
            'stage_start': [10.0],
            'stage_end': [20.0]
        })
        self.path6 = os.path.join(self.test_dir, "line6.shp")
        self.gdf6.to_file(self.path6, driver='ESRI Shapefile')

        # Test case 7: LineString with NaN stage values
        self.line7 = LineString([(0, 0), (100, 100)])
        self.gdf7 = gpd.GeoDataFrame({
            'geometry': [self.line7],
            'stage_start': [np.nan],
            'stage_end': [np.nan]
        })
        self.path7 = os.path.join(self.test_dir, "line7.shp")
        self.gdf7.to_file(self.path7, driver='ESRI Shapefile')

        # Test case 8: LineString with truncated column names
        self.line8 = LineString([(0, 0), (100, 100)])
        self.gdf8 = gpd.GeoDataFrame({
            'geometry': [self.line8],
            'stage_star': [10.0], # Truncated stage_start
            'stage_end': [20.0]  # Truncated stage_end
        })
        self.path8 = os.path.join(self.test_dir, "line8.shp")
        self.gdf8.to_file(self.path8, driver='ESRI Shapefile')


    def tearDown(self):
        # Clean up dummy shapefiles
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_load_line_features_simple(self):
        source_config = {
            "path": self.path1,
            "control_points": {
                "z_start_col": "stage_start",
                "z_end_col": "stage_end",
                "spacing": 50.0,
                "avoid_vertices": True
            }
        }
        config = {}
        cp_x, cp_y, cp_h, cp_n = load_line_features(source_config, config)

        # length is 141.42. spacing 50. num_points = floor(141.42/50) = 2.
        # distances: 25, 116.42
        self.assertEqual(len(cp_x), 2)
        self.assertEqual(len(cp_y), 2)
        self.assertEqual(len(cp_h), 2)

        self.assertAlmostEqual(cp_x[0], 25.0 * (100/141.421), 2)
        self.assertAlmostEqual(cp_y[0], 25.0 * (100/141.421), 2)
        self.assertAlmostEqual(cp_h[0], 10 + (20-10)*(25/141.421), 2)

    def test_load_line_features_multiline(self):
        source_config = {
            "path": self.path2,
            "control_points": {
                "z_start_col": "stage_start",
                "z_end_col": "stage_end",
                "spacing": 25.0,
                "avoid_vertices": True
            }
        }
        config = {}
        cp_x, cp_y, cp_h, cp_n = load_line_features(source_config, config)

        # MultiLineString: line2a (length 50*sqrt(2)), line2b (length 50*sqrt(2))
        # Total length approx 141.4
        # Spacing 25.0
        # num_points for line2a: int(141.4 / 25) = 5 (but we have two parts, so need to check logic)
        # The function iterates over parts, so each part will generate points if its length allows.
        # line2a length = sqrt(50^2 + 50^2) = 70.71
        # num_points for line2a = int(70.71 // 25) = 2
        # Distances for line2a: 12.5, 37.5
        # Stages for line2a: 15 + (25-15)*(12.5/70.71) = 16.767, 15 + (25-15)*(37.5/70.71) = 20.303
        # line2b length = sqrt(50^2 + (-50)^2) = 70.71
        # num_points for line2b = int(70.71 // 25) = 2
        # Distances for line2b: 12.5, 37.5 (relative to start of line2b)
        # Stages for line2b: 15 + (25-15)*(12.5/70.71) = 16.767, 15 + (25-15)*(37.5/70.71) = 20.303
        # Total points = 2 + 2 = 4
        # line2a length 70.71. spacing 25. num_points = floor(70.71/25) = 2.
        # line2b length 70.71. spacing 25. num_points = floor(70.71/25) = 2.
        # Total points = 4.
        self.assertEqual(len(cp_x), 4)

    def test_load_line_features_missing_stage_cols(self):
        source_config = {
            "path": self.path3,
            "control_points": {
                "z_start_col": "non_existent_start",
                "z_end_col": "non_existent_end",
                "spacing": 50.0,
                "avoid_vertices": True
            }
        }
        config = {}
        cp_x, cp_y, cp_h, cp_n = load_line_features(source_config, config)

        # Expects 0 points because missing columns skip the feature
        self.assertEqual(len(cp_x), 0)
        self.assertEqual(len(cp_y), 0)
        self.assertEqual(len(cp_h), 0)

    def test_load_line_features_empty_file(self):
        source_config = {
            "path": self.path4,
            "control_points": {
                "z_start_col": "stage_start",
                "z_end_col": "stage_end",
                "spacing": 50.0,
                "avoid_vertices": True
            }
        }
        config = {}
        cp_x, cp_y, cp_h, cp_n = load_line_features(source_config, config)

        self.assertEqual(len(cp_x), 0)
        self.assertEqual(len(cp_y), 0)
        self.assertEqual(len(cp_h), 0)

    def test_load_line_features_zero_length_line(self):
        source_config = {
            "path": self.path5,
            "control_points": {
                "z_start_col": "stage_start",
                "z_end_col": "stage_end",
                "spacing": 50.0,
                "avoid_vertices": True
            }
        }
        config = {}
        cp_x, cp_y, cp_h, cp_n = load_line_features(source_config, config)

        self.assertEqual(len(cp_x), 0)
        self.assertEqual(len(cp_y), 0)
        self.assertEqual(len(cp_h), 0)

    def test_load_line_features_spacing_larger_than_length(self):
        source_config = {
            "path": self.path6,
            "control_points": {
                "z_start_col": "stage_start",
                "z_end_col": "stage_end",
                "spacing": 100.0,
                "avoid_vertices": True
            }
        }
        config = {}
        cp_x, cp_y, cp_h, cp_n = load_line_features(source_config, config)

        # length is 70.71, spacing is 100.0. num_points = ceil(70.71/100) = 1.
        # distance = 70.71 / 2 = 35.35
        self.assertEqual(len(cp_x), 1)
        self.assertEqual(len(cp_y), 1)
        self.assertEqual(len(cp_h), 1)

    def test_load_line_features_nan_stages(self):
        source_config = {
            "path": self.path7,
            "control_points": {
                "z_start_col": "stage_start",
                "z_end_col": "stage_end",
                "spacing": 50.0,
                "avoid_vertices": True
            }
        }
        config = {}
        cp_x, cp_y, cp_h, cp_n = load_line_features(source_config, config)

        # Expects 0 points because NaN values skip the feature
        self.assertEqual(len(cp_x), 0)
        self.assertEqual(len(cp_y), 0)
        self.assertEqual(len(cp_h), 0)

    def test_load_line_features_truncated_column_names(self):
        source_config = {
            "path": self.path8,
            "control_points": {
                "z_start_col": "stage_start",
                "z_end_col": "stage_end",
                "spacing": 50.0,
                "avoid_vertices": True
            }
        }
        config = {}
        cp_x, cp_y, cp_h, cp_n = load_line_features(source_config, config)

        self.assertEqual(len(cp_x), 2)
        self.assertEqual(len(cp_y), 2)
        self.assertEqual(len(cp_h), 2)

        # Check that truncated column names were used correctly
        # length is 141.42. spacing 50. num_points = floor(141.42/50) = 2.
        # distances: 25, 116.42
        self.assertAlmostEqual(cp_x[0], 25.0 * (100/141.421), 2)
        self.assertAlmostEqual(cp_y[0], 25.0 * (100/141.421), 2)
        self.assertAlmostEqual(cp_h[0], 10 + (20-10)*(25/141.421), 2)

    def test_load_line_features_missing_path(self):
        source_config = {
            "control_points": {
                "z_start_col": "stage_start",
                "z_end_col": "stage_end",
                "spacing": 50.0
            }
        }
        config = {}
        with self.assertRaisesRegex(KeyError, "'path' must be provided"):
            load_line_features(source_config, config)

    def test_load_line_features_nonexistent_path(self):
        source_config = {
            "path": "nonexistent/path.shp",
            "control_points": {
                "z_start_col": "stage_start",
                "z_end_col": "stage_end",
                "spacing": 50.0
            }
        }
        config = {}
        with self.assertRaisesRegex(FileNotFoundError, "Line features file not found"):
            load_line_features(source_config, config)

    def test_load_line_features_invalid_spacing(self):
        source_config = {
            "path": self.path1,
            "control_points": {
                "z_start_col": "stage_start",
                "z_end_col": "stage_end",
                "spacing": 0.0,
                "avoid_vertices": True
            }
        }
        config = {}
        with self.assertRaisesRegex(ValueError, "control_point_spacing must be > 0"):
            load_line_features(source_config, config)

    def test_load_line_features_offset(self):
        source_config = {
            "path": self.path1,
            "control_points": {
                "z_start_col": "stage_start",
                "z_end_col": "stage_end",
                "spacing": 50.0,
                "avoid_vertices": True,
                "perpendicular_offset": 10.0
            }
        }
        config = {}
        cp_x, cp_y, cp_h, cp_n = load_line_features(source_config, config)
        
        # Line is (0,0) to (100,100). Tangent is (1,1) normalized to (0.707, 0.707)
        # Perpendicular is (-0.707, 0.707).
        # First point at d=25 was (17.68, 17.68)
        # With offset 10: x = 17.68 - 7.07 = 10.61, y = 17.68 + 7.07 = 24.75
        self.assertEqual(len(cp_x), 2)
        self.assertAlmostEqual(cp_x[0], 17.6776 - 7.071, 2)
        self.assertAlmostEqual(cp_y[0], 17.6776 + 7.071, 2)

    def test_load_line_features_disabled(self):
        source_config = {
            "path": self.path1,
            "control_points": {
                "enabled": False,
                "z_start_col": "stage_start",
                "z_end_col": "stage_end",
                "spacing": 50.0
            }
        }
        config = {}
        cp_x, cp_y, cp_h, cp_n = load_line_features(source_config, config)
        self.assertEqual(len(cp_x), 0)

    def test_load_line_features_include_vertices(self):
        source_config = {
            "path": self.path1,
            "control_points": {
                "z_start_col": "stage_start",
                "z_end_col": "stage_end",
                "spacing": 50.0,
                "avoid_vertices": False
            }
        }
        config = {}
        cp_x, cp_y, cp_h, cp_n = load_line_features(source_config, config)
        
        # length is 141.4. spacing 50. num_points = ceil(141.4/50) + 1 = 3 + 1 = 4
        # distances: 0, 47.14, 94.28, 141.4
        self.assertEqual(len(cp_x), 4)
        self.assertAlmostEqual(cp_x[0], 0.0)
        self.assertAlmostEqual(cp_x[-1], 100.0)

if __name__ == '__main__':
    unittest.main()
