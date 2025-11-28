#!/bin/bash

curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "prompt": "What is the capital of France?",
    "max_tokens": 50,
    "temperature": 0.7
  }'
