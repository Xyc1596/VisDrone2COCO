"""
    Check duplicated video/image/track ids.
"""

import json
from components import Dataset, DatasetDict

if __name__ == '__main__':
    coco_json = input("Dataset JSON: ")
    if coco_json:
        with open(coco_json, 'r', encoding='utf-8') as f:
            dataset_obj: DatasetDict = json.load(f)
        Dataset.fromCOCO(dataset_obj).overview()
