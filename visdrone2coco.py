import os
import cv2
import json
import time
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing.pool import Pool
from typing import List, Dict, Literal, Optional, Union, Tuple



COCOAnnotation = Dict[
    Literal["id", "category_id", "image_id", "track_id", "conf", "area", "bbox", "iscrowd"],
    Union[int, float, List[int]]
]

COCOImage = Dict[
    Literal["file_name", "id", "frame_id", "prev_frame_id", "next_frame_id",
            "video_id", "width", "height"],
    Union[str, int]
]

COCOVideo = Dict[Literal["id", "file_name"], Union[int, str]]



class Annotation:
    def __init__(self, anno: List[str]):
        x, y, w, h = [int(i) for i in anno[2:6]]    # left, top, width, height
        self.category_id = int(anno[7]) - 1
        self.image_id = int(anno[0])    # starts from 1
        self.track_id = int(anno[1])    # starts from 0
        self.conf = float(anno[6])
        self.area = w * h
        self.bbox = [x, y, w, h]

    @staticmethod
    def of(anno_line: str) -> Optional['Annotation']:
        anno = anno_line.strip()
        if len(anno) < 2:
            return None
        anno_sp = anno.split(',')
        return Annotation(anno_sp) if 0 < int(anno_sp[7]) < 11 else None

    def unserialize(self, id: int, image_id_offset: int, track_id_offset: int) -> COCOAnnotation:
        return {
            "id": id,
            "category_id": self.category_id,
            "image_id": self.image_id + image_id_offset,
            "track_id": self.track_id + track_id_offset,
            "conf": self.conf,
            "area": self.area,
            "bbox": self.bbox,
            "iscrowd": 0
        }



class Sequence:
    def __init__(self, dataset_dir: str, sequence_name: str):
        self.sequence_name = sequence_name
        self.annotations_path = os.path.join(dataset_dir, "annotations", sequence_name + ".txt")
        self.sequence_dir = os.path.join(dataset_dir, "sequences", sequence_name)
        self.img_names: List[str] = os.listdir(self.sequence_dir)

        self.img_h, self.img_w = cv2.imread(os.path.join(self.sequence_dir, self.img_names[0])).shape[:2]

        with open(self.annotations_path, 'r') as f:
            self.annotations: List[Annotation] = [
                obj for line in f.readlines() if (obj := Annotation.of(line)) is not None
            ]

    def __len__(self):
        return len(self.img_names)

    def get_num_tracks(self):
        return len(set((obj.track_id for obj in self.annotations)))

    def unserialize(
        self,
        video_id: int,
        image_id_offset: int,
        track_id_offset: int
    ) -> Tuple[COCOVideo, List[COCOImage], List[COCOAnnotation]]:
        return (
            {
                "id": video_id,
                "file_name": os.path.join("sequences", self.sequence_name)
            },
            [
                {
                    "file_name": os.path.join("sequences", self.sequence_name, img_name),
                    "id": image_id_offset + image_id,
                    "frame_id": image_id,
                    "prev_frame_id": image_id - 1 if image_id > 1 else -1,
                    "next_frame_id": image_id + 1 if image_id < len(self.img_names) else -1,
                    "video_id": video_id,
                    "width": self.img_w,
                    "height": self.img_h
                } for image_id, img_name in enumerate(self.img_names, start=1)
            ],
            [
                anno.unserialize(anno_id, image_id_offset, track_id_offset)
                for anno_id, anno in enumerate(self.annotations, start=1)
            ]
        )



class Dataset:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.sequence_names: List[str] = os.listdir(os.path.join(dataset_dir, "sequences"))
        self.unserialized: Dict[str, Tuple[COCOVideo, List[COCOImage], List[COCOAnnotation]]] = {}
        self.progress_bar = tqdm(total = len(self.sequence_names))

        self.image_id_offset = 0
        self.track_id_offset = 0

    def sequence(self, video_id: int) -> Sequence:
        return Sequence(self.dataset_dir, self.sequence_names[video_id - 1])

    def consumer(
        self,
        sequence_name: str,
        sequence: Sequence,
        out: Tuple[COCOVideo, List[COCOImage], List[COCOAnnotation]]
    ):
        self.unserialized[sequence_name] = out
        self.progress_bar.update()
        self.image_id_offset += len(sequence)
        self.track_id_offset += sequence.get_num_tracks()

    def serialize(self):
        dataset_name = os.path.split(self.dataset_dir)[-1]
        json_path = os.path.join(self.dataset_dir, "annotations", dataset_name + ".json")
        print(f"\nWriting dataset into: {json_path}")

        categories = [
            {"id":0, "name": "pedestrian"},
            {"id":1, "name": "people"},
            {"id":2, "name": "bicycle"},
            {"id":3, "name": "car"},
            {"id":4, "name": "van"},
            {"id":5, "name": "truck"},
            {"id":6, "name": "tricycle"},
            {"id":7, "name": "awning-tricycle"},
            {"id":8, "name": "bus"},
            {"id":9, "name": "motor"},
        ]
        annotations: List[COCOAnnotation] = []
        images: List[COCOImage] = []
        videos: List[COCOVideo] = []
        for sequence_name in self.sequence_names:
            annotations.extend(self.unserialized[sequence_name][2])
            images.extend(self.unserialized[sequence_name][1])
            videos.append(self.unserialized[sequence_name][0])

        with open(json_path, 'w') as f:
            json.dump(
                {
                    "categories": categories,
                    "annotations": annotations,
                    "images": images,
                    "videos": videos
                }, f
            )

if __name__ == '__main__':
    dataset_dir = input("Dataset dir (containing 2 subdirs: `annotations` & `sequences`): ")
    pool = mp.Pool(8)
    dataset = Dataset(dataset_dir)

    for video_id, sequence_name in enumerate(dataset.sequence_names, start=1):
        sequence = dataset.sequence(video_id)
        result = pool.apply_async(
            sequence.unserialize,
            args = (video_id, dataset.image_id_offset, dataset.track_id_offset),
        )
        dataset.consumer(sequence_name, sequence, result.get())

    pool.close()
    pool.join()
    dataset.progress_bar.close()

    dataset.serialize()