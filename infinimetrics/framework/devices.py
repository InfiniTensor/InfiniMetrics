from enum import Enum

class DeviceBackend(Enum):
    """
    Defines all supported hardware backends.
    The values correspond to the argument names used in argparse.
    """
    CPU = "cpu"
    NVIDIA = "nvidia"       # Corresponds to args.nvidia
    CAMBRICON = "cambricon" # Corresponds to args.cambricon
    ASCEND = "ascend"       # Corresponds to args.ascend
    ILUVATAR = "iluvatar"
    METAX = "metax"
    MOORE = "moore"
    KUNLUN = "kunlun"
    HYGON = "hygon"
    QY = "qy"
