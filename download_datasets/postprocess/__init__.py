from .delete import Delete
from .unzip import Unzip
from .untar import Untar


POSTPROCESS_MAP = {
    'Unzip': Unzip,
    'Untar': Untar,
    'Delete': Delete,
}
