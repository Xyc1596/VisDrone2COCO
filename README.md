# VisDrone2COCO

将 VisDrone **MOT** 数据集转换为 COCO 格式

开发环境：Python 3.8.20

转换前：
```
VisDrone2019-MOT-val/
 ├─ annotations
 │   ├─ uav0000086_00000_v.txt
 │   └─ ...
 └─ sequences/
     ├─ uav0000086_00000_v/
     └─ ...
```

转换后：
```
VisDrone2019-MOT-val/
 ├─ annotations
 │   ├─ uav0000086_00000_v.txt
 │   ├─ ...
 │   └─ VisDrone2019-MOT-val.json
 └─ sequences/
     ├─ uav0000086_00000_v/
     └─ ...
```