"""
Unit tests for the prompt strategies module.

Tests the PromptStrategy classes and their prompt generation,
selection logic, and performance tracking functionality.
"""

import pytest
from unittest.mock import Mock, patch

from warship_extractor.detection.prompt_strategies import (
    PromptStrategy,
    BasicPromptStrategy,
    AdaptivePromptStrategy
)


class TestBasicPromptStrategy:
    """Test cases for the BasicPromptStrategy class."""

    def test_init_default_settings(self):
        """Test BasicPromptStrategy initialization with default settings."""
        strategy = BasicPromptStrategy()
        
        assert hasattr(strategy, 'prompts')
        assert len(strategy.prompts) > 0
        assert hasattr(strategy, 'performance_tracking')

    def test_get_prompts_returns_list(self):
        """Test that get_prompts returns a list of prompts."""
        strategy = BasicPromptStrategy()
        
        prompts = strategy.get_prompts()
        
        assert isinstance(prompts, list)
        assert len(prompts) > 0
        assert all(isinstance(prompt, str) for prompt in prompts)

    def test_get_prompts_contains_warship_terms(self):
        """Test that prompts contain warship-related terms."""
        strategy = BasicPromptStrategy()
        
        prompts = strategy.get_prompts()
        
        # Check that at least some prompts contain warship-related terms
        warship_terms = ['warship', 'ship', 'vessel', 'naval', 'battleship', 'cruiser']
        
        found_terms = []
        for prompt in prompts:
            prompt_lower = prompt.lower()
            for term in warship_terms:
                if term in prompt_lower:
                    found_terms.append(term)
        
        assert len(found_terms) > 0, "No warship-related terms found in prompts"

    def test_get_primary_prompts(self):
        """Test getting primary prompts."""
        strategy = BasicPromptStrategy()
        
        primary_prompts = strategy.get_primary_prompts()
        
        assert isinstance(primary_prompts, list)
        assert len(primary_prompts) > 0
        # Primary prompts should be a subset of all prompts
        all_prompts = strategy.get_prompts()
        assert all(prompt in all_prompts for prompt in primary_prompts)

    def test_get_technical_prompts(self):
        """Test getting technical prompts."""
        strategy = BasicPromptStrategy()
        
        technical_prompts = strategy.get_technical_prompts()
        
        assert isinstance(technical_prompts, list)
        assert len(technical_prompts) > 0
        
        # Check for technical terms
        technical_terms = ['diagram', 'schematic', 'technical', 'blueprint', 'drawing']
        found_technical = False
        for prompt in technical_prompts:
            prompt_lower = prompt.lower()
            if any(term in prompt_lower for term in technical_terms):
                found_technical = True
                break
        
        assert found_technical, "No technical terms found in technical prompts"

    def test_get_historical_prompts(self):
        """Test getting historical prompts."""
        strategy = BasicPromptStrategy()
        
        historical_prompts = strategy.get_historical_prompts()
        
        assert isinstance(historical_prompts, list)
        assert len(historical_prompts) > 0

    def test_get_contextual_prompts(self):
        """Test getting contextual prompts."""
        strategy = BasicPromptStrategy()
        
        contextual_prompts = strategy.get_contextual_prompts()
        
        assert isinstance(contextual_prompts, list)
        assert len(contextual_prompts) > 0

    def test_is_ship_related_positive_cases(self):
        """Test ship-related caption detection for positive cases."""
        strategy = BasicPromptStrategy()
        
        positive_captions = [
            "A warship sailing in the ocean",
            "Naval destroyer HMS Victory",
            "Battleship in port",
            "Military vessel at sea",
            "Cruiser with cannons",
            "Submarine underwater",
            "Frigate escort ship"
        ]
        
        for caption in positive_captions:
            assert strategy.is_ship_related(caption), f"Should detect '{caption}' as ship-related"

    def test_is_ship_related_negative_cases(self):
        """Test ship-related caption detection for negative cases."""
        strategy = BasicPromptStrategy()
        
        negative_captions = [
            "A car driving on the road",
            "Beautiful landscape with mountains",
            "Person walking in the park",
            "Building architecture",
            "Text document page",
            "Empty white background"
        ]
        
        for caption in negative_captions:
            assert not strategy.is_ship_related(caption), f"Should not detect '{caption}' as ship-related"

    def test_is_ship_related_edge_cases(self):
        """Test ship-related caption detection for edge cases."""
        strategy = BasicPromptStrategy()
        
        edge_cases = [
            ("Ship model in museum", True),  # Model ships should be detected
            ("Shipping container", False),   # Shipping != ship
            ("Friendship between nations", False),  # Contains "ship" but not ship-related
            ("Naval academy building", False),  # Naval but not a ship
            ("Boat on the lake", True),     # Boats are ship-related
            ("", False),                    # Empty string
            ("   ", False)                  # Whitespace only
        ]
        
        for caption, expected in edge_cases:
            result = strategy.is_ship_related(caption)
            assert result == expected, f"Caption '{caption}' should return {expected}, got {result}"


class TestAdaptivePromptStrategy:
    """Test cases for the AdaptivePromptStrategy class."""

    def test_init_default_settings(self):
        """Test AdaptivePromptStrategy initialization."""
        strategy = AdaptivePromptStrategy()
        
        assert hasattr(strategy, 'performance_history')
        assert hasattr(strategy, 'success_threshold')
        assert hasattr(strategy, 'min_samples')
        assert len(strategy.performance_history) == 0

    def test_init_custom_settings(self):
        """Test AdaptivePromptStrategy initialization with custom settings."""
        strategy = AdaptivePromptStrategy(
            success_threshold=0.8,
            min_samples=5
        )
        
        assert strategy.success_threshold == 0.8
        assert strategy.min_samples == 5

    def test_update_performance_new_prompt(self):
        """Test updating performance for a new prompt."""
        strategy = AdaptivePromptStrategy()
        
        prompt = "Detect warships in this image"
        strategy.update_performance(prompt, success=True, confidence=0.9)
        
        assert prompt in strategy.performance_history
        assert len(strategy.performance_history[prompt]) == 1
        assert strategy.performance_history[prompt][0]['success'] is True
        assert strategy.performance_history[prompt][0]['confidence'] == 0.9

    def test_update_performance_existing_prompt(self):
        """Test updating performance for an existing prompt."""
        strategy = AdaptivePromptStrategy()
        
        prompt = "Detect warships in this image"
        
        # Add multiple performance records
        strategy.update_performance(prompt, success=True, confidence=0.9)
        strategy.update_performance(prompt, success=False, confidence=0.3)
        strategy.update_performance(prompt, success=True, confidence=0.8)
        
        assert len(strategy.performance_history[prompt]) == 3

    def test_get_prompt_success_rate_sufficient_samples(self):
        """Test getting success rate for prompt with sufficient samples."""
        strategy = AdaptivePromptStrategy(min_samples=3)
        
        prompt = "Detect warships"
        
        # Add sufficient samples
        strategy.update_performance(prompt, success=True, confidence=0.9)
        strategy.update_performance(prompt, success=False, confidence=0.3)
        strategy.update_performance(prompt, success=True, confidence=0.8)
        strategy.update_performance(prompt, success=True, confidence=0.7)
        
        success_rate = strategy.get_prompt_success_rate(prompt)
        assert success_rate == 0.75  # 3 out of 4 successful

    def test_get_prompt_success_rate_insufficient_samples(self):
        """Test getting success rate for prompt with insufficient samples."""
        strategy = AdaptivePromptStrategy(min_samples=5)
        
        prompt = "Detect warships"
        
        # Add insufficient samples
        strategy.update_performance(prompt, success=True, confidence=0.9)
        strategy.update_performance(prompt, success=False, confidence=0.3)
        
        success_rate = strategy.get_prompt_success_rate(prompt)
        assert success_rate is None

    def test_get_prompt_success_rate_unknown_prompt(self):
        """Test getting success rate for unknown prompt."""
        strategy = AdaptivePromptStrategy()
        
        success_rate = strategy.get_prompt_success_rate("Unknown prompt")
        assert success_rate is None

    def test_get_best_performing_prompts(self):
        """Test getting best performing prompts."""
        strategy = AdaptivePromptStrategy(min_samples=2)
        
        # Add performance data for multiple prompts
        prompts_data = [
            ("Good prompt", [(True, 0.9), (True, 0.8), (True, 0.85)]),  # 100% success
            ("Bad prompt", [(False, 0.2), (False, 0.3), (False, 0.1)]),  # 0% success
            ("Medium prompt", [(True, 0.7), (False, 0.4), (True, 0.6)]),  # 66% success
        ]
        
        for prompt, performances in prompts_data:
            for success, confidence in performances:
                strategy.update_performance(prompt, success=success, confidence=confidence)
        
        best_prompts = strategy.get_best_performing_prompts(limit=2)
        
        assert len(best_prompts) == 2
        assert best_prompts[0][0] == "Good prompt"  # Best should be first
        assert best_prompts[1][0] == "Medium prompt"  # Second best

    def test_get_adaptive_prompts_no_history(self):
        """Test getting adaptive prompts with no performance history."""
        strategy = AdaptivePromptStrategy()
        
        # Should fall back to basic strategy when no history
        prompts = strategy.get_adaptive_prompts()
        
        assert isinstance(prompts, list)
        assert len(prompts) > 0

    def test_get_adaptive_prompts_with_history(self):
        """Test getting adaptive prompts with performance history."""
        strategy = AdaptivePromptStrategy(min_samples=2)
        
        # Add some good performing prompts
        good_prompts = strategy.get_primary_prompts()[:3]
        for prompt in good_prompts:
            strategy.update_performance(prompt, success=True, confidence=0.9)
            strategy.update_performance(prompt, success=True, confidence=0.8)
        
        # Add some poor performing prompts
        poor_prompts = strategy.get_technical_prompts()[:2]
        for prompt in poor_prompts:
            strategy.update_performance(prompt, success=False, confidence=0.2)
            strategy.update_performance(prompt, success=False, confidence=0.3)
        
        adaptive_prompts = strategy.get_adaptive_prompts()
        
        # Should prioritize good performing prompts
        assert isinstance(adaptive_prompts, list)
        assert len(adaptive_prompts) > 0
        
        # The first few prompts should be the good performing ones
        for good_prompt in good_prompts:
            assert good_prompt in adaptive_prompts[:5]

    def test_reset_performance_history(self):
        """Test resetting performance history."""
        strategy = AdaptivePromptStrategy()
        
        # Add some performance data
        strategy.update_performance("Test prompt", success=True, confidence=0.9)
        assert len(strategy.performance_history) > 0
        
        # Reset and verify
        strategy.reset_performance_history()
        assert len(strategy.performance_history) == 0

    def test_get_performance_statistics(self):
        """Test getting performance statistics."""
        strategy = AdaptivePromptStrategy()
        
        # Add performance data
        strategy.update_performance("Prompt 1", success=True, confidence=0.9)
        strategy.update_performance("Prompt 1", success=False, confidence=0.3)
        strategy.update_performance("Prompt 2", success=True, confidence=0.8)
        
        stats = strategy.get_performance_statistics()
        
        assert 'total_prompts' in stats
        assert 'total_attempts' in stats
        assert 'overall_success_rate' in stats
        assert 'avg_confidence' in stats
        
        assert stats['total_prompts'] == 2
        assert stats['total_attempts'] == 3
        assert stats['overall_success_rate'] == 2/3

    @pytest.mark.parametrize("success_threshold,expected_threshold", [
        (0.5, 0.5),
        (0.7, 0.7),
        (0.9, 0.9),
    ])
    def test_various_success_thresholds(self, success_threshold, expected_threshold):
        """Test strategy with various success thresholds."""
        strategy = AdaptivePromptStrategy(success_threshold=success_threshold)
        
        assert strategy.success_threshold == expected_threshold

    def test_performance_tracking_timestamps(self):
        """Test that performance tracking includes timestamps."""
        strategy = AdaptivePromptStrategy()
        
        import time
        start_time = time.time()
        
        strategy.update_performance("Test prompt", success=True, confidence=0.9)
        
        end_time = time.time()
        
        record = strategy.performance_history["Test prompt"][0]
        assert 'timestamp' in record
        assert start_time <= record['timestamp'] <= end_time

    def test_max_history_limit(self):
        """Test that performance history respects maximum limit."""
        strategy = AdaptivePromptStrategy(max_history_per_prompt=3)
        
        prompt = "Test prompt"
        
        # Add more records than the limit
        for i in range(5):
            strategy.update_performance(prompt, success=True, confidence=0.8)
        
        # Should only keep the most recent records
        assert len(strategy.performance_history[prompt]) == 3

    def test_get_prompts_integration(self):
        """Test that get_prompts works correctly (integration with parent class)."""
        strategy = AdaptivePromptStrategy()
        
        prompts = strategy.get_prompts()
        
        assert isinstance(prompts, list)
        assert len(prompts) > 0
        assert all(isinstance(prompt, str) for prompt in prompts)