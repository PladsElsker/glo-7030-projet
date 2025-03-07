from .unzip import Unzip
from .untar import Untar
from .delete import Delete


POSTPROCESS_MAP = {
    'Unzip': Unzip,
    'Untar': Untar,
    'Delete': Delete,
}
