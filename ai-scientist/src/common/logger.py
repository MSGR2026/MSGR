"""
Beautiful logging system for Scientific Agent.

Provides colored console output and file logging with structured formatting.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple


class ColorFormatter(logging.Formatter):
    """Formatter with color support for terminal output."""
    
    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",      # Reset
        "BOLD": "\033[1m",       # Bold
        "DIM": "\033[2m",        # Dim
    }
    
    # Emoji icons for different levels
    ICONS = {
        "DEBUG": "🔍",
        "INFO": "✓",
        "WARNING": "⚠",
        "ERROR": "✗",
        "CRITICAL": "💥",
    }
    
    def __init__(self, use_emoji: bool = True, use_color: bool = True):
        """
        Initialize ColorFormatter.
        
        Args:
            use_emoji: Whether to use emoji icons.
            use_color: Whether to use colors (disable for non-TTY output).
        """
        super().__init__()
        self.use_emoji = use_emoji
        self.use_color = use_color and sys.stdout.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors and icons."""
        # Get color and icon
        level_name = record.levelname
        color = self.COLORS.get(level_name, self.COLORS["RESET"]) if self.use_color else ""
        reset = self.COLORS["RESET"] if self.use_color else ""
        dim = self.COLORS["DIM"] if self.use_color else ""
        icon = self.ICONS.get(level_name, "") if self.use_emoji else ""
        
        # Format timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Format level tag
        level_tag = f"[{level_name:^7}]"
        
        # Build message
        if icon:
            formatted = f"{dim}{timestamp}{reset} {color}{icon} {level_tag}{reset} {record.getMessage()}"
        else:
            formatted = f"{dim}{timestamp}{reset} {color}{level_tag}{reset} {record.getMessage()}"
        
        return formatted


class FileFormatter(logging.Formatter):
    """Formatter for file output (no colors)."""
    
    def __init__(self):
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )


class AgentLogger:
    """
    Beautiful logging system for Scientific Agent.
    
    Features:
    - Colored console output with emoji icons
    - Structured file logging
    - Section headers and separators
    - Table formatting
    - Progress indicators
    """
    
    def __init__(
        self,
        name: str,
        log_dir: str = "ai-scientist/logs",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        use_emoji: bool = True,
    ):
        """
        Initialize AgentLogger.
        
        Args:
            name: Logger name (used for log file naming).
            log_dir: Directory for log files.
            console_level: Logging level for console output.
            file_level: Logging level for file output.
            use_emoji: Whether to use emoji icons in console output.
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        self.logger.propagate = False  # Don't propagate to root logger
        
        # Console handler (with colors)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(ColorFormatter(use_emoji=use_emoji))
        self.logger.addHandler(console_handler)
        
        # File handler (plain text)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{name}_{timestamp}.log"
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(FileFormatter())
        self.logger.addHandler(file_handler)
    
    # ==================== Basic Logging ====================
    
    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs) -> None:
        """Log info message."""
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs) -> None:
        """Log error message."""
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs) -> None:
        """Log critical message."""
        self.logger.critical(msg, *args, **kwargs)
    
    # ==================== Structured Logging ====================
    
    def section(self, title: str, char: str = "─", width: int = 60) -> None:
        """
        Log a section header.
        
        Args:
            title: Section title.
            char: Character to use for separator.
            width: Total width of separator.
        """
        padding = (width - len(title) - 2) // 2
        left = char * padding
        right = char * (width - len(title) - 2 - padding)
        self.logger.info(f"\n{left} {title} {right}")
    
    def separator(self, char: str = "─", width: int = 60) -> None:
        """Log a separator line."""
        self.logger.info(char * width)
    
    def blank(self) -> None:
        """Log a blank line."""
        self.logger.info("")
    
    def header(self, title: str, width: int = 60) -> None:
        """
        Log a prominent header.
        
        Args:
            title: Header title.
            width: Total width.
        """
        border = "═" * width
        padding = (width - len(title)) // 2
        title_line = " " * padding + title
        self.logger.info(f"\n{border}")
        self.logger.info(title_line)
        self.logger.info(border)
    
    def subheader(self, title: str) -> None:
        """Log a subheader."""
        self.logger.info(f"\n### {title}")
    
    # ==================== Data Formatting ====================
    
    def table(
        self,
        headers: List[str],
        rows: List[List[Any]],
        min_width: int = 8,
    ) -> None:
        """
        Log a formatted table.
        
        Args:
            headers: Column headers.
            rows: Table rows (list of lists).
            min_width: Minimum column width.
        """
        if not rows:
            return
        
        # Calculate column widths
        col_widths = [
            max(
                min_width,
                len(str(h)),
                max(len(str(row[i])) for row in rows) if rows else 0
            )
            for i, h in enumerate(headers)
        ]
        
        # Format header
        header_line = " | ".join(
            f"{str(h):<{w}}" for h, w in zip(headers, col_widths)
        )
        separator = "-+-".join("-" * w for w in col_widths)
        
        self.logger.info(header_line)
        self.logger.info(separator)
        
        # Format rows
        for row in rows:
            row_line = " | ".join(
                f"{str(v):<{w}}" for v, w in zip(row, col_widths)
            )
            self.logger.info(row_line)
    
    def key_value(self, items: List[Tuple[str, Any]], key_width: int = 20) -> None:
        """
        Log key-value pairs.
        
        Args:
            items: List of (key, value) tuples.
            key_width: Width for key column.
        """
        for key, value in items:
            self.logger.info(f"  {key:<{key_width}}: {value}")
    
    def bullet_list(self, items: List[str], indent: int = 2) -> None:
        """
        Log a bullet list.
        
        Args:
            items: List items.
            indent: Indentation level.
        """
        prefix = " " * indent + "• "
        for item in items:
            self.logger.info(f"{prefix}{item}")
    
    def numbered_list(self, items: List[str], indent: int = 2) -> None:
        """
        Log a numbered list.
        
        Args:
            items: List items.
            indent: Indentation level.
        """
        prefix = " " * indent
        for i, item in enumerate(items, 1):
            self.logger.info(f"{prefix}{i}. {item}")
    
    # ==================== Progress Indicators ====================
    
    def step(self, step_num: int, total: int, description: str) -> None:
        """
        Log a step in a process.
        
        Args:
            step_num: Current step number.
            total: Total number of steps.
            description: Step description.
        """
        self.logger.info(f"Step {step_num}/{total}: {description}")
    
    def round_start(self, round_num: int, max_rounds: int) -> None:
        """Log the start of a round."""
        self.section(f"Round {round_num}/{max_rounds}", "─", 50)
    
    def round_end(
        self,
        round_num: int,
        success: bool,
        acc: Optional[float] = None,
        error: Optional[str] = None,
        duration: float = 0.0,
    ) -> None:
        """
        Log the end of a round.
        
        Args:
            round_num: Round number.
            success: Whether the round succeeded.
            acc: Accuracy if successful.
            error: Error message if failed.
            duration: Round duration in seconds.
        """
        if success:
            acc_str = f"{acc:.4f}" if acc is not None else "N/A"
            self.info(f"Round {round_num} completed: acc={acc_str}, duration={duration:.1f}s")
        else:
            self.error(f"Round {round_num} failed: {error}")
    
    # ==================== Session Logging ====================
    
    def session_start(
        self,
        session_id: str,
        paper_id: str,
        model_name: str,
        domain: str,
        task: str,
        max_rounds: int,
    ) -> None:
        """Log session start information."""
        self.header(f"Scientific Agent - {model_name}")
        self.key_value([
            ("Session ID", session_id),
            ("Paper ID", paper_id),
            ("Model", model_name),
            ("Domain", domain),
            ("Task", task),
            ("Max Rounds", max_rounds),
        ])
        self.separator()
    
    def session_end(
        self,
        success: bool,
        best_acc: float,
        best_round: int,
        total_rounds: int,
        duration: float,
        total_tokens: int,
        output_dir: str,
    ) -> None:
        """Log session end summary."""
        self.header("Session Complete")
        
        status = "✅ SUCCESS" if success else "❌ FAILED"
        best_acc_str = f"{best_acc:.4f}" if best_acc is not None else "N/A"
        self.key_value([
            ("Status", status),
            ("Best Accuracy", best_acc_str),
            ("Best Round", best_round),
            ("Total Rounds", total_rounds),
            ("Total Duration", f"{duration:.1f}s"),
            ("Total Tokens", f"{total_tokens:,}"),
            ("Output Dir", output_dir),
            ("Log File", str(self.log_file)),
        ])
        self.separator()
    
    # ==================== Utility ====================
    
    def get_log_file_path(self) -> Path:
        """Get the path to the log file."""
        return self.log_file
    
    def set_console_level(self, level: int) -> None:
        """Set console output level."""
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                handler.setLevel(level)
    
    def set_file_level(self, level: int) -> None:
        """Set file output level."""
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.setLevel(level)
    
    def close(self) -> None:
        """Close all handlers and release resources."""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


def create_logger(
    name: str,
    log_dir: str = "ai-scientist/logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    use_emoji: bool = True,
) -> AgentLogger:
    """
    Factory function to create an AgentLogger.
    
    Args:
        name: Logger name.
        log_dir: Directory for log files.
        console_level: Console logging level.
        file_level: File logging level.
        use_emoji: Whether to use emoji icons.
        
    Returns:
        Configured AgentLogger instance.
    """
    return AgentLogger(
        name=name,
        log_dir=log_dir,
        console_level=console_level,
        file_level=file_level,
        use_emoji=use_emoji,
    )

