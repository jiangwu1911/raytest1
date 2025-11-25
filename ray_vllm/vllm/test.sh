#!/bin/bash

curl -X POST http://192.168.1.217:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer fake-key" \
     -d '{
           "model": "qwen-0.5b",
           "messages": [{"role": "user", "content": "Hello, tell me the captial of British!"}]
         }'
