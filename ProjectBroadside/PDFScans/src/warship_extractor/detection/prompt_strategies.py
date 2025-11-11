"""
Comprehensive prompt strategies for detecting warship illustrations using Florence-2.

This module provides various prompt strategies optimized for different types
of warship illustrations found in historical naval documents.
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class PromptType(Enum):
    """Types of detection prompts available."""
    OBJECT_DETECTION = "OD"
    DENSE_REGION_CAPTION = "DENSE_REGION_CAPTION"
    CAPTION = "CAPTION"
    DETAILED_CAPTION = "DETAILED_CAPTION"


class WarshipPromptStrategy:
    """
    Manages different prompt strategies for warship detection.
    """
    
    def __init__(self):
        """Initialize the prompt strategy manager."""
        self._primary_prompts = self._build_primary_prompts()
        self._technical_prompts = self._build_technical_prompts()
        self._historical_prompts = self._build_historical_prompts()
        self._contextual_prompts = self._build_contextual_prompts()
        
        logger.info("WarshipPromptStrategy initialized with comprehensive prompt sets")
    
    def _build_primary_prompts(self) -> List[str]:
        """Build primary warship detection prompts."""
        return [
            "<OD>warship illustration",
            "<OD>naval vessel diagram", 
            "<OD>battleship drawing",
            "<OD>warship schematic",
            "<OD>ship illustration",
            "<OD>naval ship",
            "<OD>military vessel",
            "<OD>war vessel"
        ]
    
    def _build_technical_prompts(self) -> List[str]:
        """Build technical drawing and blueprint prompts."""
        return [
            "<OD>ship blueprint",
            "<OD>naval architecture drawing",
            "<OD>vessel cross-section",
            "<OD>ship profile view",
            "<OD>ship side view",
            "<OD>ship plan view",
            "<OD>naval engineering diagram",
            "<OD>ship construction drawing",
            "<OD>vessel technical drawing",
            "<OD>ship deck plan",
            "<OD>hull design"
        ]
    
    def _build_historical_prompts(self) -> List[str]:
        """Build prompts for historical warship types."""
        return [
            "<OD>destroyer illustration",
            "<OD>frigate diagram",
            "<OD>submarine drawing",
            "<OD>cruiser illustration",
            "<OD>dreadnought battleship",
            "<OD>torpedo boat",
            "<OD>gunboat illustration",
            "<OD>ironclad ship",
            "<OD>armored cruiser",
            "<OD>monitor warship",
            "<OD>corvette illustration",
            "<OD>steam warship",
            "<OD>sailing warship"
        ]
    
    def _build_contextual_prompts(self) -> List[str]:
        """Build contextual prompts for comprehensive detection."""
        return [
            "<DENSE_REGION_CAPTION>",
            "<OD>naval illustration", 
            "<OD>maritime vessel",
            "<OD>armed ship",
            "<OD>combat vessel",
            "<OD>naval architecture",
            "<OD>ship design",
            "<OD>vessel diagram"
        ]
    
    def get_all_prompts(self) -> List[str]:
        """Get all available prompts."""
        return (
            self._primary_prompts + 
            self._technical_prompts + 
            self._historical_prompts + 
            self._contextual_prompts
        )
    
    def get_primary_prompts(self) -> List[str]:
        """Get primary detection prompts (most reliable)."""
        return self._primary_prompts.copy()
    
    def get_technical_prompts(self) -> List[str]:
        """Get technical drawing prompts."""
        return self._technical_prompts.copy()
    
    def get_historical_prompts(self) -> List[str]:
        """Get historical warship type prompts."""
        return self._historical_prompts.copy()
    
    def get_contextual_prompts(self) -> List[str]:
        """Get contextual detection prompts."""
        return self._contextual_prompts.copy()
    
    def get_prompts_by_strategy(self, strategy: str) -> List[str]:
        """
        Get prompts by strategy name.
        
        Args:
            strategy: Strategy name ('primary', 'technical', 'historical', 'contextual', 'all')
            
        Returns:
            List of prompts for the specified strategy
        """
        strategy_map = {
            'primary': self.get_primary_prompts,
            'technical': self.get_technical_prompts,
            'historical': self.get_historical_prompts,
            'contextual': self.get_contextual_prompts,
            'all': self.get_all_prompts
        }
        
        if strategy.lower() not in strategy_map:
            logger.warning(f"Unknown strategy '{strategy}', using primary prompts")
            return self.get_primary_prompts()
        
        return strategy_map[strategy.lower()]()
    
    def get_optimized_prompt_set(self, max_prompts: Optional[int] = None) -> List[str]:
        """
        Get an optimized set of prompts for balanced coverage.
        
        Args:
            max_prompts: Maximum number of prompts to return
            
        Returns:
            Optimized prompt list
        """
        # Start with most reliable prompts
        optimized = [
            "<OD>warship illustration",
            "<OD>battleship drawing", 
            "<OD>naval vessel diagram",
            "<OD>ship blueprint",
            "<OD>destroyer illustration",
            "<OD>vessel cross-section",
            "<DENSE_REGION_CAPTION>",
            "<OD>military vessel",
            "<OD>naval architecture drawing",
            "<OD>submarine drawing"
        ]
        
        # Add more if requested
        if max_prompts is None or max_prompts > len(optimized):
            additional = [
                "<OD>cruiser illustration",
                "<OD>ship profile view",
                "<OD>frigate diagram",
                "<OD>naval ship",
                "<OD>torpedo boat"
            ]
            optimized.extend(additional)
        
        # Limit to requested count
        if max_prompts is not None:
            optimized = optimized[:max_prompts]
        
        logger.debug(f"Generated optimized prompt set with {len(optimized)} prompts")
        return optimized
    
    def filter_ship_related_captions(self, captions: List[str]) -> List[str]:
        """
        Filter caption results for ship-related content.
        
        Args:
            captions: List of caption strings
            
        Returns:
            Filtered captions containing ship-related keywords
        """
        ship_keywords = {
            'ship', 'vessel', 'boat', 'naval', 'warship', 'battleship',
            'destroyer', 'frigate', 'cruiser', 'submarine', 'gunboat',
            'torpedo', 'fleet', 'navy', 'maritime', 'hull', 'sail',
            'cannon', 'gun', 'armor', 'deck', 'bridge', 'mast',
            'anchor', 'bow', 'stern', 'port', 'starboard', 'dreadnought',
            'ironclad', 'corvette', 'monitor', 'steam', 'turret'
        }
        
        filtered = []
        for caption in captions:
            caption_lower = caption.lower()
            if any(keyword in caption_lower for keyword in ship_keywords):
                filtered.append(caption)
        
        logger.debug(f"Filtered {len(filtered)} ship-related captions from {len(captions)} total")
        return filtered
    
    def get_prompt_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about available prompts.
        
        Returns:
            Dictionary with prompt statistics and information
        """
        return {
            "total_prompts": len(self.get_all_prompts()),
            "primary_prompts": len(self._primary_prompts),
            "technical_prompts": len(self._technical_prompts),
            "historical_prompts": len(self._historical_prompts),
            "contextual_prompts": len(self._contextual_prompts),
            "prompt_types": [pt.value for pt in PromptType],
            "strategies": ["primary", "technical", "historical", "contextual", "all"]
        }


class AdvancedPromptStrategy:
    """
    Advanced prompt strategy with dynamic optimization and context awareness.
    """
    
    def __init__(self, base_strategy: Optional[WarshipPromptStrategy] = None):
        """
        Initialize advanced strategy.
        
        Args:
            base_strategy: Base strategy to extend (creates new if None)
        """
        self.base_strategy = base_strategy or WarshipPromptStrategy()
        self.performance_history: Dict[str, float] = {}
        self.context_weights: Dict[str, float] = {}
        
    def update_prompt_performance(self, prompt: str, success_rate: float) -> None:
        """
        Update performance tracking for a prompt.
        
        Args:
            prompt: The prompt string
            success_rate: Success rate (0.0 to 1.0)
        """
        self.performance_history[prompt] = success_rate
        logger.debug(f"Updated performance for '{prompt}': {success_rate:.2f}")
    
    def get_adaptive_prompts(
        self, 
        document_context: Optional[str] = None,
        max_prompts: int = 10
    ) -> List[str]:
        """
        Get adaptively selected prompts based on performance and context.
        
        Args:
            document_context: Context about the document type
            max_prompts: Maximum number of prompts to return
            
        Returns:
            Adaptively selected prompt list
        """
        all_prompts = self.base_strategy.get_all_prompts()
        
        # Score prompts based on performance history
        scored_prompts = []
        for prompt in all_prompts:
            base_score = 0.5  # Default score
            
            # Use performance history if available
            if prompt in self.performance_history:
                base_score = self.performance_history[prompt]
            
            # Adjust score based on document context
            context_bonus = self._calculate_context_bonus(prompt, document_context)
            final_score = min(1.0, base_score + context_bonus)
            
            scored_prompts.append((prompt, final_score))
        
        # Sort by score and return top prompts
        scored_prompts.sort(key=lambda x: x[1], reverse=True)
        selected = [prompt for prompt, score in scored_prompts[:max_prompts]]
        
        logger.info(f"Selected {len(selected)} adaptive prompts")
        return selected
    
    def _calculate_context_bonus(self, prompt: str, context: Optional[str]) -> float:
        """Calculate context-based bonus for a prompt."""
        if not context:
            return 0.0
        
        context_lower = context.lower()
        prompt_lower = prompt.lower()
        
        # Bonus for matching document type indicators
        bonuses = {
            'blueprint': 0.2 if 'blueprint' in prompt_lower else 0.0,
            'technical': 0.15 if any(word in prompt_lower for word in ['technical', 'diagram', 'plan']) else 0.0,
            'historical': 0.1 if any(word in prompt_lower for word in ['destroyer', 'battleship', 'cruiser']) else 0.0,
            'jane': 0.2 if 'warship' in prompt_lower else 0.0,  # Jane's Fighting Ships context
        }
        
        total_bonus = 0.0
        for keyword, bonus in bonuses.items():
            if keyword in context_lower:
                total_bonus += bonus
        
        return min(0.3, total_bonus)  # Cap bonus at 0.3


# Global instance for easy access
default_strategy = WarshipPromptStrategy()