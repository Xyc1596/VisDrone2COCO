import pathlib
from components import Dataset, DatasetType

if __name__ == '__main__':
    dataset_dir = input("Dataset dir (containing 2 subdirs: `annotations` & `sequences`): ")
    if dataset_dir:
        presets_path = pathlib.Path(__file__).resolve().parent / "presets.toml"
        dataset_type = DatasetType.fromPreset(presets_path, "VisDrone")
        Dataset(dataset_type.CATEGORIES).setStartIds(dataset_type) \
                                        .loadFromVisDrone(dataset_dir) \
                                        .json(indent=2)
