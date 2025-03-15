from .delete import Delete
from .untar import Untar
from .unzip import Unzip

POSTPROCESS_MAP = {
    "Unzip": Unzip,
    "Untar": Untar,
    "Delete": Delete,
}
