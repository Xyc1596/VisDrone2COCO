import os
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, TypedDict
from collections import defaultdict, OrderedDict

from .image import Image, ImageDict
from .annotation import Annotation


class VideoDict(TypedDict):
    id: int
    file_name: str


class Video:
    VIDEO_ID_START: int = 1

    def __init__(self, id: int, file_name: str):
        # e.g. file_name = "sequences\uav0000086_00000_v"
        self.id = id
        self.file_name = file_name

        self.__images: OrderedDict[int, Image] = OrderedDict()  # indexed by image_id

    def __len__(self):
        return len(self.__images)

    def __getitem__(self, image_id: int) -> Image:
        return self.__images[image_id]

    @property
    def images(self) -> List[Image]:
        return list(self.__images.values())

    @property
    def image_ids(self) -> List[int]:
        return list(self.__images.keys())

    @property
    def track_ids(self) -> List[int]:
        return list(OrderedDict.fromkeys(
            track_id for image in self.__images.values() for track_id in image.track_ids
        ))

    @property
    def num_annotations(self) -> int:
        return sum(len(image) for image in self.__images.values())

    @classmethod
    def fromCOCO(cls, obj: VideoDict) -> 'Video':
        return cls(id=obj["id"], file_name=obj["file_name"])

    @classmethod
    def fromDir(cls, id: int, path: str) -> 'Video':
        return cls(id, os.path.join(Path(path).parts[-2:]))


    def addImageFromCOCO(self, obj: ImageDict) -> None:
        self.__images[obj["id"]] = Image.fromCOCO(obj)

    def loadFromVisDrone(
        self,
        dataset_dir: str,
        image_num_in_other_videos: int = 0,
        track_num_in_other_videos: int = 0
    ) -> 'Video':
        # ----- Get image h & w
        image_base_names = os.listdir((video_dir := os.path.join(dataset_dir, self.file_name)))
        img0 = cv2.imread(os.path.join(video_dir, image_base_names[0]))
        height, width = img0.shape[:2]

        # ----- Load annotations
        annotation_path = os.path.join(dataset_dir, "annotations", os.path.basename(self.file_name) + ".txt")
        annotations: Dict[int, List[Annotation]] = defaultdict(list)    # divide by image_id
        with open(annotation_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if (annotation := Annotation.fromVisDrone(
                    line, image_num_in_other_videos, track_num_in_other_videos
                )) is not None:
                    annotations[annotation.image_id].append(annotation)

        # ----- Load image & distribute annotations
        for frame_id, img_base_name in enumerate(image_base_names, Image.FRAME_ID_START):
            image_id = image_num_in_other_videos + frame_id
            self.__images[image_id] = Image(
                file_name = os.path.join(self.file_name, img_base_name),
                id = image_id,
                frame_id = frame_id,
                video_id = self.id,
                width = width,
                height = height
            ).withAnnotations(annotations[image_id])

        return self


    def dict(self) -> Tuple[VideoDict, List[ImageDict], List[Annotation]]:
        image_dicts = []
        annotation_dicts = []
        MAX_FRAME_ID = list(self.__images.values())[-1].frame_id
        for image in self.__images.values():
            image_dict_out = image.dict(MAX_FRAME_ID)
            image_dicts.append(image_dict_out[0])
            annotation_dicts.extend(image_dict_out[1])

        return {
            "id": self.id,
            "file_name": self.file_name
        }, image_dicts, annotation_dicts
