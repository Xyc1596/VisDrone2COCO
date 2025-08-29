from typing import TypedDict, List, Sequence, Tuple
from collections import OrderedDict

from .annotation import Annotation, AnnotationDict


class ImageDict(TypedDict):
    """
    * `id`: unique; starts from 1
    * `frame_id`: can repeat in different sequences; starts from 1
    * `video_id`: the `id` property of the sequence; starts from 1
    """
    file_name: str
    id: int
    frame_id: int
    prev_frame_id: int
    next_frame_id: int
    video_id: int
    width: int
    height: int


class Image:
    FRAME_ID_START: int = 1

    def __init__(
        self,
        file_name: str,
        id: int,
        frame_id: int,
        video_id: int,
        width: int,
        height: int
    ):
        self.file_name = file_name
        self.id = id
        self.frame_id = frame_id
        self.video_id = video_id
        self.width = width
        self.height = height

        self.__annotations: OrderedDict[int, Annotation] = OrderedDict()    # indexed by annotation_id

    def __len__(self):
        return len(self.__annotations)

    def __getitem__(self, annotation_id: int) -> Annotation:
        return self.__annotations[annotation_id]

    @property
    def annotations(self) -> List[Annotation]:
        return list(self.__annotations.values())

    @property
    def track_ids(self) -> List[int]:
        return [annotation.track_id for annotation in self.__annotations.values()]


    @classmethod
    def fromCOCO(cls, obj: ImageDict) -> 'Image':
        return cls(
            file_name=obj["file_name"],
            id = obj["id"],
            frame_id=obj["frame_id"],
            video_id=obj["video_id"],
            width=obj["width"],
            height=obj["height"]
        )


    def addAnnotationFromCOCO(self, obj: AnnotationDict) -> None:
        self.__annotations[obj["id"]] = Annotation.fromCOCO(obj)

    def withAnnotations(self, annotations: Sequence[Annotation]) -> 'Image':
        for id, annotation in enumerate(annotations, Annotation.ANNOTATION_ID_START):
            self.__annotations[id] = annotation.withId(id)
        return self


    def dict(self, max_frame_id_in_video: int) -> Tuple[ImageDict, List[AnnotationDict]]:
        return {
            "file_name": self.file_name,
            "id": self.id,
            "frame_id": self.frame_id,
            "prev_frame_id": self.frame_id-1 if self.frame_id > self.FRAME_ID_START else -1,
            "next_frame_id": self.frame_id+1 if self.frame_id < max_frame_id_in_video else -1,
            "video_id": self.video_id,
            "width": self.width,
            "height": self.height
        }, [
            annotation.dict() for annotation in self.__annotations.values()
        ]
