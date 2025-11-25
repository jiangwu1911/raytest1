#!/bin/bash

curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer FAKE_KEY" \
  -H "Content-Type: application/json" \
  -d '{ "model": "my-qwen-VL", "messages": [ { "role": "user", "content": [ {"type": "text", "text": "What do you see in this image?"}, {"type": "image_url", "image_url": { "url": "http://images.cocodataset.org/val2017/000000039769.jpg" }} ] } ] }'
