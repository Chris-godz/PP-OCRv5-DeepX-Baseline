from enum import Enum, IntEnum, auto

class SessionType(str, Enum):
    onnxruntime = "OnnxRuntime"
    simulator = "Simulator"
    dxruntime = "DxRuntime"


class EvaluationType(str, Enum):
    image_classification = "ImageClassification"
    coco = "ObjectDection"
    segmentation = "ImageSegmentation"
    voc = "ObjectDetection"
    bsd68 = "ImageDenosing"
    widerface = "FaceDetection"
    omnidoc = "OmniDocBench"

    def metric(self) -> str:
        if self.value == EvaluationType.image_classification:
            return "TopK1, TopK5"
        elif self.value == EvaluationType.coco:
            return "mAP, mAP50"
        elif self.value == EvaluationType.voc:
            return "mAP50"
        elif self.value == EvaluationType.segmentation:
            return "mIoU"
        elif self.value == EvaluationType.widerface:
            return "AP"
        else:
            raise ValueError(f"Invalid Evaluation Type value. {self.value}")


class DatasetType(str, Enum):
    imagenet = "ImageNet"
    coco = "COCO"
    voc_seg = "VOCSegmentation"
    voc_od = "VOC2007Detection"
    bsd68 = "BSD68"
    city = "CitySpace"
    widerface = "WiderFace"
    omnidoc = "OmniDocBench"


class ResizeMode(str, Enum):
    torchvision = auto()
    default = auto()
    pad = auto()
    pycls = auto()
    ocr = auto()
    
    @classmethod
    def has_value(cls, value: str) -> bool:
        return value in cls._value2member_map_


class ResizeArgEnum(str, Enum):
    size = auto()
    interpolation = auto()
    backend = auto()
    align_side = auto()
    scale_method = auto()
    pad_location = auto()
    pad_value = auto()


class BackendEnum(str, Enum):
    cv2 = auto()
    pil = auto()


class AlignSideEnum(str, Enum):
    both = auto()
    long = auto()
    short = auto()


class ScaleMethodEnum(str, Enum):
    scale_up = auto()
    scale_down = auto()


class InterpolationEnum(str, Enum):
    BILINEAR = "BILINEAR"
    LINEAR = "LINEAR"
    NEAREST = "NEAREST"
    BICUBIC = "BICUBIC"

    @classmethod
    def has_value(cls, value: str) -> bool:
        return value in cls._value2member_map_


class PILResizeInterpolationEnum(IntEnum):
    NEAREST = 0
    LANCZOS = 1
    BILINEAR = 2
    LINEAR = 2
    BICUBIC = 3
    BOX = 4
    HAMMING = 5

    def __repr__(self) -> str:
        return self.name


class CVResizeInterpolationEnum(IntEnum):
    NEAREST = 0
    LINEAR = 1
    BILINEAR = 1
    CUBIC = 2
    AREA = 3
    LANCZOS4 = 4
