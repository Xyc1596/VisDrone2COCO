import math
from enum import Enum
from typing import Tuple, List, Optional


class Align(Enum):
    LEFT = 0
    CENTER = 1
    RIGHT = 2


class TableData:
    def __init__(self, content: Optional[str], bold: bool = False, color: int = 39):
        self.content = "" if content is None else content
        self.control_code = f"\033[{1 if bold else ''};{color}m"

    @property
    def width(self):
        return len(str(self.content))

    def toString(self, width: int = -1, align: Align = Align.LEFT) -> str:
        num_spaces = width-self.width
        if num_spaces < 0:
            lspaces = 0
            rspaces = 0
        elif align == Align.LEFT:
            lspaces = 0
            rspaces = num_spaces
        elif align == Align.RIGHT:
            lspaces = num_spaces
            rspaces = 0
        else:
            lspaces = math.floor(num_spaces/2)
            rspaces = num_spaces - lspaces
        return f"{self.control_code}{' ' * lspaces}{self.content}{' ' * rspaces}\033[0m"


class Table:
    def __init__(self, shape: Tuple[int, int], title: str = ""):
        self.shape = shape
        self.__items: List[List[TableData]] = [
            [None] * self.shape[1] for _ in range(self.shape[0])
        ]
        self.__title = title

    def setItem(self, row: int, col: int, data: TableData):
        self.__items[row][col] = data

    def setHeadRow(self, *contents: Optional[str]) -> 'Table':
        for idx, content in enumerate(contents[:self.shape[1]]):
            self.__items[0][idx] = TableData(content, True)
        return self

    def setDataRow(self, row: int, *contents: Optional[str]) -> 'Table':
        for idx, content in enumerate(contents[:self.shape[1]]):
            self.__items[row][idx] = TableData(content, idx==0)
        return self

    def toString(self) -> str:
        _widths = [[0] * self.shape[0] for _ in range(self.shape[1])]
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                _widths[col][row] = self.__items[row][col].width
        widths = [max(w) for w in _widths]

        output_cols = 2 * self.shape[1] - 1
        outputs = [[" \u2502 "] * output_cols for _ in range(self.shape[0])]
        for row, row_items in enumerate(self.__items):
            for col, item in enumerate(row_items):
                if item is None:
                    item = TableData("")
                outputs[row][2 * col] = item.toString(widths[col])

        sep = ["\u2500\u253c\u2500"] * output_cols
        for col in range(self.shape[1]):
            sep[2 * col] = "\u2500" * widths[col]
        outputs.insert(1, sep)

        if len(self.__title) > 0:
            title = f" \033[1;30;42m {self.__title} \033[0m "
            total_width = sum(widths) + 3 * (self.shape[1] - 1)
            line_width = total_width - len(self.__title) - 4
            lwidth = math.floor(line_width / 2)
            rwidth = line_width - lwidth
            outputs.insert(0, ["\n" + "\u2500"*lwidth + title + "\u2500"*rwidth])

        return "\n".join("".join(line) for line in outputs) + "\n"
