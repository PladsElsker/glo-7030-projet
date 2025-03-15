from .delete import Delete
from .unzip import Unzip

POSTPROCESS_MAP = {
    "Unzip": Unzip,
    "Delete": Delete,
}
