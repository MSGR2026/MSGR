"""
Code Extractor for Scientific Agent.

Extracts code blocks from LLM responses.
"""
from __future__ import annotations

import re
from typing import Optional


class CodeExtractor:
    """
    Extracts code and YAML blocks from LLM responses.
    
    Supports multiple formats:
    - ```python ... ```
    - ```yaml ... ```
    - ```code ... ```
    - Raw code blocks
    """
    
    # Patterns for code block extraction
    PYTHON_PATTERN = re.compile(
        r'```(?:python|py)?\s*\n(.*?)```',
        re.DOTALL | re.IGNORECASE
    )
    
    YAML_PATTERN = re.compile(
        r'```(?:yaml|yml)?\s*\n(.*?)```',
        re.DOTALL | re.IGNORECASE
    )
    
    GENERIC_CODE_PATTERN = re.compile(
        r'```(?:\w+)?\s*\n(.*?)```',
        re.DOTALL
    )
    
    def extract_python(self, response: str) -> str:
        """
        Extract Python code from LLM response.
        
        Args:
            response: Raw LLM response text.
            
        Returns:
            Extracted Python code, or empty string if not found.
        """
        if not response:
            return ""
        
        # Try Python-specific pattern first
        matches = self.PYTHON_PATTERN.findall(response)
        if matches:
            # Return the longest match (usually the main code block)
            return max(matches, key=len).strip()
        
        # Try generic code block
        matches = self.GENERIC_CODE_PATTERN.findall(response)
        if matches:
            # Filter out YAML blocks and return longest Python-like block
            python_matches = [
                m for m in matches
                if self._looks_like_python(m)
            ]
            if python_matches:
                return max(python_matches, key=len).strip()
            # Fallback to longest generic block
            return max(matches, key=len).strip()
        
        return ""
    
    def extract_yaml(self, response: str) -> str:
        """
        Extract YAML content from LLM response.
        
        Args:
            response: Raw LLM response text.
            
        Returns:
            Extracted YAML content, or empty string if not found.
        """
        if not response:
            return ""
        
        # Try YAML-specific pattern first
        matches = self.YAML_PATTERN.findall(response)
        if matches:
            return max(matches, key=len).strip()
        
        # Try generic code block and check if it looks like YAML
        matches = self.GENERIC_CODE_PATTERN.findall(response)
        if matches:
            yaml_matches = [
                m for m in matches
                if self._looks_like_yaml(m)
            ]
            if yaml_matches:
                return max(yaml_matches, key=len).strip()
        
        return ""
    
    def clean_code(self, code: str) -> str:
        """
        Clean extracted code.
        
        - Remove leading/trailing whitespace
        - Remove common prefixes/suffixes
        - Normalize line endings
        
        Args:
            code: Raw extracted code.
            
        Returns:
            Cleaned code.
        """
        if not code:
            return ""
        
        # Normalize line endings
        code = code.replace('\r\n', '\n').replace('\r', '\n')
        
        # Strip whitespace
        code = code.strip()
        
        # Remove common markdown artifacts
        lines = code.split('\n')
        
        # Remove leading empty lines
        while lines and not lines[0].strip():
            lines.pop(0)
        
        # Remove trailing empty lines
        while lines and not lines[-1].strip():
            lines.pop()
        
        return '\n'.join(lines)
    
    def _looks_like_python(self, code: str) -> bool:
        """Check if code looks like Python."""
        python_indicators = [
            'import ', 'from ', 'def ', 'class ', 'self.',
            'if __name__', 'return ', 'for ', 'while ',
            'torch.', 'nn.', 'np.', 'import torch',
        ]
        return any(indicator in code for indicator in python_indicators)
    
    def _looks_like_yaml(self, code: str) -> bool:
        """Check if code looks like YAML."""
        lines = code.strip().split('\n')
        if not lines:
            return False
        
        # YAML typically has key: value patterns
        yaml_pattern = re.compile(r'^[\s-]*[\w_]+\s*:')
        yaml_lines = sum(1 for line in lines if yaml_pattern.match(line))
        
        # If more than 30% of non-empty lines look like YAML, it's probably YAML
        non_empty = [l for l in lines if l.strip()]
        if non_empty:
            return yaml_lines / len(non_empty) > 0.3
        
        return False


# Convenience function
def extract_code(response: str, language: str = "python") -> str:
    """
    Convenience function to extract code.
    
    Args:
        response: LLM response text.
        language: "python" or "yaml".
        
    Returns:
        Extracted code.
    """
    extractor = CodeExtractor()
    if language.lower() in ("yaml", "yml"):
        return extractor.clean_code(extractor.extract_yaml(response))
    return extractor.clean_code(extractor.extract_python(response))

