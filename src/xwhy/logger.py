"""Global logging configuration."""

import logging

logger = logging.getLogger("xwhy")

logger.addHandler(logging.NullHandler())
