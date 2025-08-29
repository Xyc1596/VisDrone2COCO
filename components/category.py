from typing import TypedDict, List, Sequence


class CategoryDict(TypedDict):
    id: int
    name: str


class Category:
    CATEGORY_ID_START: int = 1

    def __init__(self, id: int, name: str):
        self.id = id
        self.name = name

    def dict(self) -> CategoryDict:
        return {
            "id": self.id,
            "name": self.name
        }

    @classmethod
    def __getCategories(cls, names: Sequence[str]) -> List[CategoryDict]:
        return [Category(id_, name).dict() for id_, name in enumerate(names, start=cls.CATEGORY_ID_START)]

    @classmethod
    def getVisDroneCategories(cls) -> List[CategoryDict]:
        return cls.__getCategories((
            "pedestrian", "people", "bicycle", "car", "van",
            "truck", "tricycle", "awning-tricycle", "bus", "motor"
        ))
