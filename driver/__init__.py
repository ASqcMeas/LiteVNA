from .E5080B import VNA_E5080B
from .dummy import VNA_DUMMY
from .VNA import VNA
from .ZNB import VNA_ZNB20
def get_VNA( address, model=None )->VNA:
    # Normalize model name to uppercase for case-insensitive matching.
    # Without this, 'znb' or 'e5080b' would silently fall through to VNA_DUMMY.
    model_upper = model.upper() if model is not None else None
    match model_upper:
        case "E5080B":
            return VNA_E5080B(address)
        case "ZNB":
            return VNA_ZNB20(address)
        case _:
            return VNA_DUMMY(address)