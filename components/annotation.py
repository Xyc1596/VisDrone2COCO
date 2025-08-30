from typing import TypedDict, List, Optional

from .dataset_type import DatasetType


class AnnotationDict(TypedDict):
    """
    * `id`: unique; starts from 1
    * `image_id`: the `id` property of the image; starts from 1
    * `track_id`: unique; starts from 0 (VisDrone) / 1 (COCO)
    * `bbox`: [t, l, w, h]
    """
    id: int
    category_id: int
    image_id: int
    track_id: int
    area: int
    bbox: List[int]
    iscrowd: int


class Annotation:
    ANNOTATION_ID_START: int = 1
    TRACK_ID_START: int = 1
    CATEGORY_ID_START: int = 0

    def __init__(
        self,
        id: int,
        image_id: int,
        track_id: int,
        bbox: List[int],
        score: int,
        category_id: int
    ):
        self.id = id
        self.image_id = image_id            # [0]
        self.track_id = track_id            # [1]
        self.bbox = bbox                    # [2:6], tlwh
        self.score = score                  # [6], 0 or 1
        self.category_id = category_id      # [7]


    @classmethod
    def fromVisDrone(
        cls,
        id: int,
        line: str,
        image_num_in_other_videos: int = 0,
        track_num_in_other_videos: int = 0
    ) -> 'Annotation':
        anno_sp = line.split(',')
        return cls(
            id=id,
            image_id=int(anno_sp[0]) + image_num_in_other_videos,
            track_id=int(anno_sp[1]) + track_num_in_other_videos,
            bbox=[int(i) for i in anno_sp[2:6]],
            score=int(anno_sp[6]),
            category_id=int(anno_sp[7]) + cls.CATEGORY_ID_START - 1
        )

    @classmethod
    def fromCOCO(cls, obj: AnnotationDict) -> 'Annotation':
        return cls(
            id = obj["id"],
            image_id=obj["image_id"],
            track_id=obj["track_id"],
            bbox=obj["bbox"],
            score=1,
            category_id=obj["category_id"]
        )


    @property
    def isValid(self) -> bool:
        """
        Exclude annotations with category 0 (ignored regions) / 11 (others) or with score 0
        """
        return 0 <= self.category_id - self.CATEGORY_ID_START < 10 and self.score > 0


    def dict(self) -> AnnotationDict:
        return {
            "id": self.id,
            "category_id": self.category_id,
            "image_id": self.image_id,
            "track_id": self.track_id,
            "area": self.bbox[2] * self.bbox[3],
            "bbox": self.bbox,
            "iscrowd": 0
        }
