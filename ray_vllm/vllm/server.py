import os
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

serve.start(
    http_options={
        "host": "0.0.0.0",
        "port": 8000
    }
)

llm_config = LLMConfig(
    model_loading_config={
        "model_id": "qwen-0.5b",
        "model_source": "Qwen/Qwen2.5-1.5B-Instruct",
    },
    deployment_config={
        "autoscaling_config": {
            "min_replicas": 0,
            "max_replicas": 1,
        }
    },
    engine_kwargs={
        "tensor_parallel_size": 1,
    }
)

app = build_openai_app({"llm_configs": [llm_config]})
serve.run(app, blocking=True)
