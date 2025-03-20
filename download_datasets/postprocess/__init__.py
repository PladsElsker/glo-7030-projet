from .delete import Delete
from .unpack import Unpack

POSTPROCESS_MAP = {
    "Unpack": Unpack,
    "Delete": Delete,
}
