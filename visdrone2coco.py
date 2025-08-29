from components import Dataset, Category


if __name__ == '__main__':
    dataset_dir = input("Dataset dir (containing 2 subdirs: `annotations` & `sequences`): ")
    if dataset_dir:
        Dataset.setStartIds(1, 1, 1, 1, 0)
        Dataset(Category.getVisDroneCategories()).loadFromVisDrone(dataset_dir).json()
