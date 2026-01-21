import logging
import sys
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

# Custom theme for the logger
custom_theme = Theme({
    "logging.level.debug": "dim white",
    "logging.level.info": "bold blue",
    "logging.level.warning": "bold yellow",
    "logging.level.error": "bold red",
    "logging.level.critical": "bold white on red",
})

# Create console with custom theme
console = Console(theme=custom_theme, stderr=True)

# Configure RichHandler with enhanced options
rich_handler = RichHandler(
    console=console,
    rich_tracebacks=True,
    show_path=True,
    show_time=True,
    show_level=True,
    markup=True,
    tracebacks_show_locals=False,
    tracebacks_extra_lines=2,
    tracebacks_theme="monokai",
    log_time_format="[%H:%M:%S]",
)

# Configure root logger to WARNING to avoid spam from dependencies
logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%H:%M:%S]",
    handlers=[rich_handler],
    force=True,  # Override any existing configuration
)

# Create custom logger for lejurobot with INFO level
logger = logging.getLogger("lejurobot")
logger.setLevel(logging.INFO)

# Prevent propagation to root logger to avoid duplicate messages
logger.propagate = False

# Add handler to lejurobot logger if it doesn't have one
if not logger.handlers:
    logger.addHandler(rich_handler)

if __name__ == "__main__":
    logger.debug("This is a debug message")
    logger.info("Starting the application...")
    logger.warning("This is a warning!")
    logger.error("Something went wrong!")
    logger.critical("Critical error occurred!")