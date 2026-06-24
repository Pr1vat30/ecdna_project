import os
import logging
from typing import Optional

def setup_logging(level: int = logging.INFO) -> None:
    """Configures the package-wide logging format and level."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def save_report(content: str, file_path: str) -> None:
    """
    Saves text or markdown report content to a file.
    Automatically creates parent directories if they do not exist.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    logging.getLogger(__name__).info(f"Report saved to {file_path}")
