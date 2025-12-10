import json
import os

test_data = [
    {
        "id": "001",
        "image": "test_images/cat01.jpeg",  
        "conversations": [
            {"from": "user", "value": "<img>test_images/cat01.jpeg</img>请识别图片中的物体"},
            {"from": "assistant", "value": "图片中的物体是: 小橘猫"}
        ]
    },
    {
        "id": "002",
        "image": "test_images/cat02.jpeg",  
        "conversations": [
            {"from": "user", "value": "<img>test_images/cat02.jpeg</img>请识别图片中的物体"},
            {"from": "assistant", "value": "图片中的物体是: 小狸花猫"}
        ]
    },
    {
        "id": "003",
        "image": "test_images/example.jpg",  
        "conversations": [
            {"from": "user", "value": "<img>test_images/example.jpg</img>请识别图片中的物体"},
            {"from": "assistant", "value": "图片中的物体是: 人"}
        ]
    },
    {
        "id": "004",
        "image": "test_images/panda.jpeg",  
        "conversations": [
            {"from": "user", "value": "<img>test_images/panda.jpeg</img>请识别图片中的物体"},
            {"from": "assistant", "value": "图片中的物体是: 熊猫"}
        ]
    },
    {
        "id": "005",
        "image": "test_images/win10.jpg",  
        "conversations": [
            {"from": "user", "value": "<img>test_images/win10.jpg</img>请识别图片中的物体"},
            {"from": "assistant", "value": "图片中的物体是: win10开机画面"}
        ]
    }
]

with open("qwen_vl_finetune_data.jsonl", "w", encoding="utf-8") as f:
    for item in test_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
