from modelscope.hub.snapshot_download import snapshot_download

def download_model():
    model_dir = snapshot_download('unsloth/Qwen3-VL-4B-Instruct', cache_dir='./qwen3-vl-4b')

download_model()

