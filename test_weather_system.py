"""
Unit Tests for Fuzzy Weather Comfort System
Tests all major components and edge cases
"""

import unittest
import numpy as np
import sys
import os

# Import the fuzzy system
from fuzzy_weather_system import predict_comfort, temperature, humidity, comfort_index


class TestInputBounds(unittest.TestCase):
    """Test input boundary conditions"""
    
    def test_temperature_below_minimum(self):
        """Test temperature below -10°C gets clipped"""
        score, level = predict_comfort(-20, 50)
        self.assertIsNotNone(score)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
    
    def test_temperature_above_maximum(self):
        """Test temperature above 50°C gets clipped"""
        score, level = predict_comfort(60, 50)
        self.assertIsNotNone(score)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
    
    def test_humidity_below_minimum(self):
        """Test humidity below 0% gets clipped"""
        score, level = predict_comfort(25, -10)
        self.assertIsNotNone(score)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
    
    def test_humidity_above_maximum(self):
        """Test humidity above 100% gets clipped"""
        score, level = predict_comfort(25, 150)
        self.assertIsNotNone(score)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
    
    def test_boundary_minimum_temperature(self):
        """Test exact minimum temperature (-10°C)"""
        score, level = predict_comfort(-10, 50)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
    
    def test_boundary_maximum_temperature(self):
        """Test exact maximum temperature (50°C)"""
        score, level = predict_comfort(50, 50)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)


class TestOutputBounds(unittest.TestCase):
    """Test that outputs are always valid"""
    
    def test_output_always_in_range(self):
        """Test 1000 random inputs for valid outputs"""
        np.random.seed(42)
        for _ in range(1000):
            temp = np.random.uniform(-20, 60)
            humid = np.random.uniform(-10, 110)
            
            score, level = predict_comfort(temp, humid)
            
            self.assertGreaterEqual(score, 0, f"Score {score} below 0")
            self.assertLessEqual(score, 100, f"Score {score} above 100")
            self.assertIn(level, ['Uncomfortable', 'Neutral', 'Comfortable'])
    
    def test_comfort_level_consistency(self):
        """Test that comfort level matches score thresholds"""
        test_cases = [
            (30, 'Uncomfortable'),  # Below 40
            (50, 'Neutral'),        # Between 40-70
            (80, 'Comfortable'),    # Above 70
        ]
        
        for expected_score, expected_level in test_cases:
            # Create conditions that should produce this score
            if expected_score < 40:
                temp, humid = 5, 80  # Cold and humid
            elif expected_score < 70:
                temp, humid = 30, 70  # Warm and humid
            else:
                temp, humid = 22, 45  # Ideal conditions
            
            score, level = predict_comfort(temp, humid)
            
            # Check level is appropriate for score
            if score < 40:
                self.assertEqual(level, 'Uncomfortable')
            elif score < 70:
                self.assertEqual(level, 'Neutral')
            else:
                self.assertEqual(level, 'Comfortable')


class TestComfortScenarios(unittest.TestCase):
    """Test realistic weather scenarios"""
    
    def test_ideal_comfort_conditions(self):
        """Test ideal comfort: mild temp, moderate humidity"""
        score, level = predict_comfort(22, 50)
        self.assertGreater(score, 60, "Expected high comfort for ideal conditions")
        self.assertIn(level, ['Neutral', 'Comfortable'])
    
    def test_cold_winter_uncomfortable(self):
        """Test cold conditions are uncomfortable"""
        score, level = predict_comfort(0, 40)
        self.assertLess(score, 50, "Expected low comfort for cold conditions")
    
    def test_hot_humid_uncomfortable(self):
        """Test hot and humid is very uncomfortable"""
        score, level = predict_comfort(38, 90)
        self.assertLess(score, 50, "Expected low comfort for hot+humid")
        self.assertEqual(level, 'Uncomfortable')
    
    def test_pleasant_spring_day(self):
        """Test pleasant spring conditions"""
        score, level = predict_comfort(20, 45)
        self.assertGreater(score, 60, "Expected high comfort for spring day")
    
    def test_dry_heat_better_than_humid_heat(self):
        """Test that dry heat is more comfortable than humid heat"""
        score_dry, _ = predict_comfort(35, 25)
        score_humid, _ = predict_comfort(35, 80)
        self.assertGreater(score_dry, score_humid, 
                          "Dry heat should be more comfortable than humid heat")


class TestMonotonicity(unittest.TestCase):
    """Test expected monotonic relationships"""
    
    def test_temperature_optimal_around_mild(self):
        """Test that comfort peaks at mild temperatures"""
        temps = [-5, 5, 15, 22, 28, 35, 42]
        scores = [predict_comfort(t, 50)[0] for t in temps]
        
        # Find peak
        max_idx = scores.index(max(scores))
        max_temp = temps[max_idx]
        
        # Peak should be in mild range (15-28°C)
        self.assertGreaterEqual(max_temp, 15)
        self.assertLessEqual(max_temp, 28)
    
    def test_extreme_humidity_reduces_comfort(self):
        """Test that extreme humidity (very low or very high) reduces comfort"""
        # At comfortable temperature
        temp = 22
        
        humidities = [10, 30, 50, 70, 90]
        scores = [predict_comfort(temp, h)[0] for h in humidities]
        
        # Extreme values (10%, 90%) should be less comfortable than moderate
        mid_score = scores[2]  # 50% humidity
        
        # At least one extreme should be less comfortable
        self.assertTrue(scores[0] < mid_score or scores[4] < mid_score)


class TestConsistency(unittest.TestCase):
    """Test system consistency and determinism"""
    
    def test_deterministic_output(self):
        """Test that same inputs always produce same outputs"""
        temp, humid = 25, 60
        
        results = [predict_comfort(temp, humid) for _ in range(10)]
        
        # All scores should be identical
        scores = [r[0] for r in results]
        self.assertEqual(len(set(scores)), 1, "System should be deterministic")
        
        # All levels should be identical
        levels = [r[1] for r in results]
        self.assertEqual(len(set(levels)), 1, "System should be deterministic")
    
    def test_small_change_small_effect(self):
        """Test that small input changes produce small output changes"""
        base_score, _ = predict_comfort(25, 60)
        
        # Small perturbations
        perturbed_scores = [
            predict_comfort(25.1, 60)[0],
            predict_comfort(25, 60.5)[0],
            predict_comfort(24.9, 60)[0],
            predict_comfort(25, 59.5)[0],
        ]
        
        for perturbed in perturbed_scores:
            diff = abs(perturbed - base_score)
            self.assertLess(diff, 10, 
                           "Small input change should not cause large output change")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and unusual inputs"""
    
    def test_all_minimum_inputs(self):
        """Test both inputs at minimum"""
        score, level = predict_comfort(-10, 0)
        self.assertIsNotNone(score)
        self.assertIsInstance(level, str)
    
    def test_all_maximum_inputs(self):
        """Test both inputs at maximum"""
        score, level = predict_comfort(50, 100)
        self.assertIsNotNone(score)
        self.assertIsInstance(level, str)
    
    def test_zero_humidity(self):
        """Test very dry conditions (0% humidity)"""
        score, level = predict_comfort(25, 0)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
    
    def test_full_humidity(self):
        """Test saturated air (100% humidity)"""
        score, level = predict_comfort(25, 100)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)


class TestMembershipFunctions(unittest.TestCase):
    """Test membership function properties"""
    
    def test_membership_values_in_range(self):
        """Test that all membership values are between 0 and 1"""
        # Test temperature memberships
        for t in np.linspace(-10, 50, 100):
            for term in ['cold', 'mild', 'warm', 'hot']:
                membership = fuzz.interp_membership(temperature.universe, 
                                                   temperature[term].mf, t)
                self.assertGreaterEqual(membership, 0)
                self.assertLessEqual(membership, 1)
    
    def test_at_least_one_membership_nonzero(self):
        """Test that at any point, at least one membership is > 0"""
        import skfuzzy as fuzz
        
        for t in np.linspace(-10, 50, 50):
            memberships = []
            for term in ['cold', 'mild', 'warm', 'hot']:
                membership = fuzz.interp_membership(temperature.universe, 
                                                   temperature[term].mf, t)
                memberships.append(membership)
            
            self.assertGreater(max(memberships), 0, 
                             f"At temp {t}, no membership is active")


class TestRuleCoverage(unittest.TestCase):
    """Test that rules cover all important scenarios"""
    
    def test_all_temperature_categories_covered(self):
        """Test predictions for all temperature categories"""
        test_temps = [0, 18, 26, 40]  # Cold, Mild, Warm, Hot
        
        for temp in test_temps:
            for humid in [20, 50, 80]:
                score, level = predict_comfort(temp, humid)
                self.assertIsNotNone(score, 
                                   f"No output for temp={temp}, humid={humid}")
    
    def test_all_humidity_categories_covered(self):
        """Test predictions for all humidity categories"""
        test_humids = [20, 45, 70, 90]  # Low, Moderate, High, Very High
        
        for humid in test_humids:
            for temp in [5, 22, 35]:
                score, level = predict_comfort(temp, humid)
                self.assertIsNotNone(score, 
                                   f"No output for temp={temp}, humid={humid}")


def run_all_tests():
    """Run all test suites with detailed output"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestInputBounds,
        TestOutputBounds,
        TestComfortScenarios,
        TestMonotonicity,
        TestConsistency,
        TestEdgeCases,
        TestMembershipFunctions,
        TestRuleCoverage
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run:  {result.testsRun}")
    print(f"Successes:  {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures:   {len(result.failures)}")
    print(f"Errors:     {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ ALL TESTS PASSED!")
        print("="*70)
    else:
        print("\n✗ SOME TESTS FAILED")
        print("="*70)
    
    return result


if __name__ == '__main__':
    import skfuzzy as fuzz
    
    print("\n" + "="*70)
    print("FUZZY WEATHER COMFORT SYSTEM - UNIT TESTS")
    print("="*70)
    
    run_all_tests()