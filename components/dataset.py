import os
import json
from tqdm import tqdm
from typing import Dict, List, Optional, TypedDict
from collections import OrderedDict, Counter, defaultdict

from .video import VideoDict, Video
from .image import ImageDict, Image
from .annotation import AnnotationDict, Annotation
from .category import CategoryDict, Category


class DatasetDict(TypedDict):
    categories: List[CategoryDict]
    annotations: List[AnnotationDict]
    images: List[ImageDict]
    videos: List[VideoDict]


class Dataset:
    def __init__(self, categories: List[CategoryDict], dataset_dir: Optional[str] = None):
        self.__categories = categories
        self.__videos: OrderedDict[int, Video] = OrderedDict()  # indexed by video_id
        self.__dataset_dir = dataset_dir

    def __len__(self):
        return len(self.__videos)

    def __getitem__(self, video_id: int) -> Video:
        return self.__videos[video_id]

    @property
    def video_ids(self) -> List[int]:
        return list(self.__videos.keys())

    @property
    def image_ids_per_video(self) -> Dict[int, List[int]]:
        return {video_id: video.image_ids for video_id, video in self.__videos.items()}

    @property
    def image_ids(self) -> List[int]:
        return [image_id for video in self.__videos.values() for image_id in video.image_ids]

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

        # ----- Check duplicated ids -----
        duplicated_video_ids = {id_: num for id_, num in Counter(video_ids_loaded).items() if num > 1}
        duplicated_image_ids = {id_: num for id_, num in Counter(image_ids_loaded).items() if num > 1}
        track_ids_loaded = (id_ for ids_ in track_ids_loaded_per_video.values() for id_ in ids_)
        duplicated_track_ids = {id_: num for id_, num in Counter(track_ids_loaded).items() if num > 1}
        if len(duplicated_video_ids) > 0:
            print(f"[WARNING] Duplicated video ids found ({len(duplicated_video_ids)} total)")
        if len(duplicated_image_ids) > 0:
            print(f"[WARNING] Duplicated image ids found ({len(duplicated_image_ids)} total)")
        if len(duplicated_track_ids) > 0:
            print(f"[WARNING] Duplicated track ids found ({len(duplicated_track_ids)} total)")

        return instance

    def loadFromVisDrone(self, dataset_dir: str) -> 'Dataset':
        self.__dataset_dir = os.path.abspath(dataset_dir)
        annotations_dir = os.path.join(self.__dataset_dir, "annotations")
        image_num_in_other_videos = 0
        track_num_in_other_videos = 0

        files = (file for file in os.listdir(annotations_dir) if file.endswith(".txt"))
        for video_id, file in tqdm(enumerate(files, Video.VIDEO_ID_START), desc="Loading videos"):
            seq_name, ext = os.path.splitext(file)
            if ext != ".txt":
                continue

            video = Video(video_id, os.path.join("sequences", seq_name)).loadFromVisDrone(
                dataset_dir, image_num_in_other_videos, track_num_in_other_videos
            )
            self.__videos[video_id] = video
            image_num_in_other_videos += len(video)
            track_num_in_other_videos = max(video.track_ids) + 1 - Annotation.TRACK_ID_START

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
        video_ids = self.video_ids
        image_ids = self.image_ids
        track_ids = [track_id for video in self.__videos.values() for track_id in video.track_ids]
        print(
            "-------- OVERVIEW --------\n"
            "[Videos]\n"
            f"    Total video num: {len(video_ids)}\n"
            f"    Min video id: {min(video_ids)}\n"
            f"    Max video id: {max(video_ids)}\n"
            "[Images]\n"
            f"    Total image num: {len(image_ids)}\n"
            f"    Min image id: {min(image_ids)}\n"
            f"    Max image id: {max(image_ids)}\n"
            "[Annotations]\n"
            f"    Total annotation num: {sum(video.num_annotations for video in self.__videos.values())}\n"
            f"    Total track num: {len(track_ids)}\n"
            f"    Min track id: {min(track_ids)}\n"
            f"    Max track id: {max(track_ids)}"
        )


    @staticmethod
    def setStartIds(
        category_id_start: int = 1,
        video_id_start: int = 1,
        frame_id_start: int = 1,
        annotation_id_start: int = 1,
        track_id_start: int = 0
    ):
        Category.CATEGORY_ID_START = category_id_start
        Video.VIDEO_ID_START = video_id_start
        Image.FRAME_ID_START = frame_id_start
        Annotation.ANNOTATION_ID_START = annotation_id_start
        Annotation.TRACK_ID_START = track_id_start
