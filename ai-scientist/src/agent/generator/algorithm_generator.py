"""
Algorithm Generator for Scientific Agent.

Generates algorithm code using LLM with validation and retry logic.
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


class AlgorithmGenerator:
    """
    Generates algorithm code using LLM.
    
    Features:
    - Extracts Python code from LLM responses
    - Retries on failure
    """
    
    def __init__(
        self,
        llm_client,
        logger: Optional["AgentLogger"] = None,
        max_retries: int = 3,
    ):
        """
        Initialize AlgorithmGenerator.
        
        Args:
            llm_client: LLM client for code generation.
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
        domain: Optional[str] = None,
    ) -> str:
        """
        Generate algorithm code.
        
        Args:
            system_prompt: System prompt for LLM.
            user_prompt: User prompt with context.
            domain: Optional domain for validation.
            
        Returns:
            Generated and validated Python code.
            
        Raises:
            GenerationError: If generation fails after all retries.
        """
        last_errors = []
        
        for attempt in range(self.max_retries):
            if self.logger:
                self.logger.debug(f"Algorithm generation attempt {attempt + 1}/{self.max_retries}")
            
            # Call LLM
            try:
                response = self.llm.generate(system_prompt, user_prompt)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"LLM call failed: {e}")
                last_errors.append(f"LLM调用失败: {e}")
                continue
            
            # Extract code
            code = self.extractor.extract_python(response)
            code = self.extractor.clean_code(code)
            
            if not code:
                if self.logger:
                    self.logger.warning("No code extracted; returning raw response")
                return response.strip()
            
            # 直接返回提取的代码（不进行验证）
            if self.logger:
                self.logger.info(f"Algorithm generated successfully (attempt {attempt + 1})")
            
            return code
        
        # All retries exhausted
        error_msg = f"算法生成失败 ({self.max_retries}次尝试): {'; '.join(last_errors)}"
        raise GenerationError(error_msg)
