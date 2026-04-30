MODEL_CONFIG = {
    'model_path': 'yolov8n.pt',
    'conf_threshold': 0.5,
    'iou_threshold': 0.45,
    'device': 'cuda',
    'backend': 'pytorch'
}

CLASS_NAME_MAP = {
    'person': '行人',
    'car': '车辆',
    'bicycle': '自行车',
    'motorcycle': '摩托车',
    'bus': '公交车',
    'truck': '卡车',
    'boat': '船',
    'traffic light': '交通灯',
    'stop sign': '停车标志',
    'bench': '长椅',
    'bird': '鸟',
    'cat': '猫',
    'dog': '狗',
    'horse': '马',
    'sheep': '羊',
    'cow': '牛',
    'elephant': '大象',
    'bear': '熊',
    'zebra': '斑马',
    'giraffe': '长颈鹿',
}