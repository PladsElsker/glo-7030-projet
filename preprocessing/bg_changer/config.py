from pathlib import Path
from typing import Any, Final

import numpy as np

LOWER_GREEN = np.array([40, 50, 50])
UPPER_GREEN = np.array([90, 255, 255])

DEFAULT_CONFIG: Final[dict[str, Any]] = {
    "background_dir_path": Path(__file__).parent.parent.parent / "data" / "backgrounds" / "pictures",
}
