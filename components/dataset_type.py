import tomli
from typing import Sequence, List, TypedDict


class CategoryDict(TypedDict):
    id: int
    name: str


class PresetDict(TypedDict):
    video_id_start: int
    frame_id_start: int
    track_id_start: int
    annotation_id_start: int
    category_id_start: int
    category_names: Sequence[str]


class DatasetType:
    def __init__(
        self,
        video_id_start: int,
        frame_id_start: int,
        track_id_start: int,
        annotation_id_start: int,
        category_id_start: int,
        category_names: Sequence[str]
    ):
        self.__video_id_start = video_id_start
        self.__frame_id_start = frame_id_start
        self.__track_id_start = track_id_start
        self.__annotation_id_start = annotation_id_start
        self.__category_id_start = category_id_start
        self.__category_names = category_names

    @property
    def VIDEO_ID_START(self) -> int:
        return self.__video_id_start
    @property
    def FRAME_ID_START(self) -> int:
        return self.__frame_id_start
    @property
    def TRACK_ID_START(self) -> int:
        return self.__track_id_start
    @property
    def ANNOTATION_ID_START(self) -> int:
        return self.__annotation_id_start
    @property
    def CATEGORY_ID_START(self) -> int:
        return self.__category_id_start
    @property
    def CATEGORY_NAMES(self) -> Sequence[str]:
        return self.__category_names

    @property
    def CATEGORIES(self) -> List[CategoryDict]:
        return [
            {"id": self.__category_id_start + idx, "name": name}
            for idx, name in enumerate(self.__category_names)
        ]

    @staticmethod
    def fromPreset(file: str, preset_name: str) -> 'DatasetType':
        with open(file, 'rb') as f:
            obj: PresetDict = tomli.load(f)[preset_name]
        return DatasetType(
            video_id_start=obj["video_id_start"],
            frame_id_start=obj["frame_id_start"],
            track_id_start=obj["track_id_start"],
            annotation_id_start=obj["annotation_id_start"],
            category_id_start=obj["category_id_start"],
            category_names=obj["category_names"]
        )
