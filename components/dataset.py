import os
import json
from tqdm import tqdm
from typing import Dict, List, Optional, TypedDict, Tuple, Iterable
from collections import OrderedDict, Counter, defaultdict

from .video import VideoDict, Video
from .image import ImageDict, Image
from .annotation import AnnotationDict, Annotation
from .dataset_type import CategoryDict, DatasetType

from .utils import Table


class DatasetDict(TypedDict):
    categories: List[CategoryDict]
    annotations: List[AnnotationDict]
    images: List[ImageDict]
    videos: List[VideoDict]


class Dataset:
    def __init__(self, categories: List[CategoryDict]):
        self.__categories = categories
        self.__videos: OrderedDict[int, Video] = OrderedDict()  # indexed by video_id
        self.__dataset_dir = ""

    def __len__(self):
        return len(self.__videos)

    def __getitem__(self, video_id: int) -> Video:
        return self.__videos[video_id]

    @property
    def image_ids_per_video(self) -> Dict[int, List[int]]:
        return {video_id: video.image_ids for video_id, video in self.__videos.items()}

    @property
    def dataset_name(self) -> str:
        return os.path.basename(self.__dataset_dir)


    def addVideoFromCOCO(self, obj: VideoDict) -> None:
        self.__videos[obj["id"]] = Video.fromCOCO(obj)

    @classmethod
    def fromCOCO(cls, obj: DatasetDict) -> 'Dataset':
        instance = cls(obj["categories"])
        video_ids_loaded = []
        image_ids_loaded = []
        track_ids_loaded_per_video = defaultdict(set)
        annotation_ids_loaded = []

        # ----- Videos -----
        for video_obj in tqdm(obj["videos"], desc="Loading videos"):
            instance.addVideoFromCOCO(video_obj)
            video_ids_loaded.append(video_obj["id"])

        # ----- Images -----
        for image_obj in tqdm(obj["images"], desc="Loading images"):
            instance[image_obj["video_id"]].addImageFromCOCO(image_obj)
            image_ids_loaded.append(image_obj["id"])

        # ----- Annotations -----
        video_id_of_images = {
            image_id: video_id for video_id, image_ids in instance.image_ids_per_video.items()
                               for image_id in image_ids
        }
        for annotation_obj in tqdm(obj["annotations"], desc="Loading annotations"):
            image_id = annotation_obj["image_id"]
            video_id = video_id_of_images[image_id]
            instance[video_id][image_id].addAnnotationFromCOCO(annotation_obj)
            track_ids_loaded_per_video[video_id].add(annotation_obj["track_id"])
            annotation_ids_loaded.append(annotation_obj["id"])

        # ----- Check duplicated ids -----
        def check_duplicated_ids(name: str, ids: Iterable[int]):
            duplicated_ids = {id_: num for id_, num in Counter(ids).items() if num > 1}
            if len(duplicated_ids) > 0:
                print(f"\033[30;43;1m[WARNING]\033[33;49m Duplicated {name} ids found "
                      f"({len(duplicated_ids)} total)\033[0m")

        check_duplicated_ids("video", video_ids_loaded)
        check_duplicated_ids("image", image_ids_loaded)
        check_duplicated_ids(
            "track", (id_ for ids_ in track_ids_loaded_per_video.values() for id_ in ids_)
        )
        check_duplicated_ids("annotations", annotation_ids_loaded)

        return instance

    def loadFromVisDrone(self, dataset_dir: str) -> 'Dataset':
        self.__dataset_dir = os.path.abspath(dataset_dir)
        annotations_dir = os.path.join(self.__dataset_dir, "annotations")
        image_num_in_other_videos = 0
        track_num_in_other_videos = 0
        anno_num_in_other_videos = 0

        files = (file for file in os.listdir(annotations_dir) if file.endswith(".txt"))
        for (video_id, file) in tqdm(enumerate(files, Video.VIDEO_ID_START), desc="Loading videos"):
            seq_name, ext = os.path.splitext(file)
            if ext != ".txt":
                continue

            video = Video(video_id, os.path.join("sequences", seq_name)).loadFromVisDrone(
                dataset_dir,
                image_num_in_other_videos,
                track_num_in_other_videos,
                anno_num_in_other_videos
            )
            self.__videos[video_id] = video
            image_num_in_other_videos += len(video)
            track_num_in_other_videos = max(video.all_track_ids) + 1 - Annotation.TRACK_ID_START
            anno_num_in_other_videos = max(video.all_annotation_ids) + 1 - Annotation.ANNOTATION_ID_START

        return self


    def dict(self) -> DatasetDict:
        video_dicts = []
        image_dicts = []
        annotation_dicts = []
        for video in self.__videos.values():
            video_dict_out = video.dict()
            video_dicts.append(video_dict_out[0])
            image_dicts.extend(video_dict_out[1])
            annotation_dicts.extend(video_dict_out[2])

        return {
            "categories": self.__categories,
            "annotations": annotation_dicts,
            "images": image_dicts,
            "videos": video_dicts
        }

    def json(self, json_path: Optional[str] = None, indent: Optional[int] = None) -> None:
        if json_path is None:
            json_path = os.path.join(self.__dataset_dir, "annotations", self.dataset_name+".json")
        print("\nWriting dataset into: " + json_path)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.dict(), f, indent=indent)


    def overview(self):
        def make_row(name: str, ids: List[int]) -> Tuple[str, int, int, int]:
            return name, len(ids), min(ids), max(ids)

        video_ids = []
        image_ids = []
        track_ids = []
        all_track_ids = []
        annotation_ids = []
        all_annotation_ids = []
        for (video_id, video) in tqdm(self.__videos.items(), desc="Analysing videos"):
            video_ids.append(video_id)
            image_ids.extend(video.image_ids)
            track_ids.extend(video.track_ids)
            all_track_ids.extend(video.all_track_ids)
            annotation_ids.extend(video.annotation_ids)
            all_annotation_ids.extend(video.all_annotation_ids)

        metrics = {
            "TOTAL NUM": lambda ids: len(ids),
            "MIN ID": lambda ids: min(ids),
            "MAX ID": lambda ids: max(ids)
        }
        data = {
            "VIDEOS": video_ids,
            "IMAGES": image_ids,
            "TRACKS": track_ids,
            "ALL TRACKS": all_track_ids,
            "ANNOTATIONS": annotation_ids,
            "ALL_ANNOTATIONS": all_annotation_ids
        }
        table = Table((4, 7), "OVERVIEW").setHeadRow(None, *data.keys())
        for idx, (metric, func) in enumerate(metrics.items(), 1):
            table.setDataRow(idx, metric, *(func(ids) for ids in data.values()))
        print(table.toString())


    def setStartIds(self, dataset_type: DatasetType) -> 'Dataset':
        Video.VIDEO_ID_START = dataset_type.VIDEO_ID_START
        Image.FRAME_ID_START = dataset_type.FRAME_ID_START
        Annotation.TRACK_ID_START = dataset_type.TRACK_ID_START
        Annotation.ANNOTATION_ID_START = dataset_type.ANNOTATION_ID_START
        Annotation.CATEGORY_ID_START = dataset_type.CATEGORY_ID_START
        return self
