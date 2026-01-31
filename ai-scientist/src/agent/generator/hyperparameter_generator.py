"""
Hyperparameter Generator for Scientific Agent.

Generates YAML hyperparameter configurations using LLM with retry logic.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, TYPE_CHECKING

# Add src to path for imports
SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent.generator.extractor import CodeExtractor

if TYPE_CHECKING:
    from common.logger import AgentLogger


class GenerationError(Exception):
    """Exception raised when code generation fails."""
    pass


class HyperparameterGenerator:
    """
    Generates hyperparameter configurations using LLM.
    
    Features:
    - Extracts YAML content from LLM responses
    - Retries on failure
    """
    
    def __init__(
        self,
        llm_client,
        logger: Optional["AgentLogger"] = None,
        max_retries: int = 3,
    ):
        """
        Initialize HyperparameterGenerator.
        
        Args:
            llm_client: LLM client for generation.
            logger: Optional logger for debugging.
            max_retries: Maximum number of generation retries.
        """
        self.llm = llm_client
        self.logger = logger
        self.max_retries = max_retries
        self.extractor = CodeExtractor()
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Generate hyperparameter configuration.
        
        Args:
            system_prompt: System prompt for LLM.
            user_prompt: User prompt with context.
            
        Returns:
            Generated YAML configuration.
            
        Raises:
            GenerationError: If generation fails after all retries.
        """
        last_errors = []
        
        for attempt in range(self.max_retries):
            if self.logger:
                self.logger.debug(f"Hyperparameter generation attempt {attempt + 1}/{self.max_retries}")
            
            # Call LLM
            try:
                response = self.llm.generate(system_prompt, user_prompt)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"LLM call failed: {e}")
                last_errors.append(f"LLM调用失败: {e}")
                continue
            
            # Extract YAML
            yaml_content = self.extractor.extract_yaml(response)
            yaml_content = self.extractor.clean_code(yaml_content)
            
            if not yaml_content:
                if self.logger:
                    self.logger.warning("No YAML extracted; returning raw response")
                return response.strip()
            
            # 直接返回提取的 YAML (不进行验证)
            if self.logger:
                self.logger.info(f"Hyperparameters generated successfully (attempt {attempt + 1})")
            
            return yaml_content
        
        # All retries exhausted
        error_msg = f"超参数生成失败 ({self.max_retries}次尝试): {'; '.join(last_errors)}"
        raise GenerationError(error_msg)
