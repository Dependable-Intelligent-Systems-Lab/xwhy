---
title: Configure XWhy Logging
description: Enable structured XWhy progress logs in Python, Jupyter Notebook, or Google Colab.
---

# Configure logging

XWhy uses the standard Python logger named `xwhy`.

```python
import logging
import sys

logger = logging.getLogger("xwhy")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
```

The handler guard prevents duplicate log lines when a notebook cell is executed repeatedly.
