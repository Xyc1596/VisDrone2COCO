from typing import TypedDict, List, Optional, Sequence

from .category import Category


class AnnotationDict(TypedDict):
    """
    * `id`: can repeat in different images; starts from 1
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

    def __init__(
        self,
        id: Optional[int],
        image_id: int,
        track_id: int,
        bbox: Sequence[int],
        category_id: int
    ):
        self.id = id
        self.image_id = image_id            # [0]
        self.track_id = track_id            # [1]
        self.bbox = bbox                    # [2:6], tlwh
        self.category_id = category_id      # [7]


    @classmethod
    def fromVisDrone(
        cls,
        line: str,
        image_num_in_other_videos: int = 0,
        track_num_in_other_videos: int = 0
    ) -> Optional['Annotation']:
        anno = line.strip()
        if len(anno) < 2:
            return cls(None)
        anno_sp = anno.split(',')
        cid = int(anno_sp[7])

        # Exclude annotations with category 0 (ignore_regions) / 11 (others) or with score 0
        return cls(
            id=None,
            image_id=int(anno_sp[0]) + image_num_in_other_videos,
            track_id=int(anno_sp[1]) + track_num_in_other_videos,
            bbox=[int(i) for i in anno_sp[2:6]],
            category_id=cid + Category.CATEGORY_ID_START - 1
        ) if 0 < cid < 11 and int(anno_sp[6]) > 0 else None

    @classmethod
    def fromCOCO(cls, obj: AnnotationDict) -> 'Annotation':
        return cls(
            id = obj["id"],
            image_id=obj["image_id"],
            track_id=obj["track_id"],
            bbox=obj["bbox"],
            category_id=obj["category_id"]
        )

    def withId(self, id: int) -> 'Annotation':
        self.id = id
        return self


    def dict(self) -> AnnotationDict:
        assert self.id is not None
        return {
            "id": self.id,
            "category_id": self.category_id,
            "image_id": self.image_id,
            "track_id": self.track_id,
            "area": self.bbox[2] * self.bbox[3],
            "bbox": self.bbox,
            "iscrowd": 0
        }
