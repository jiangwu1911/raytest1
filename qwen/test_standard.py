"""
测试标准微调脚本的环境依赖
"""
import sys
import subprocess

def check_package(package_name):
    """检查包是否已安装"""
    try:
        __import__(package_name)
        print(f"✓ {package_name} 已安装")
        return True
    except ImportError:
        print(f"✗ {package_name} 未安装")
        return False

def main():
    print("检查环境依赖...")
    
    required_packages = [
        "torch",
        "transformers",
        "datasets",
        "peft",
        "PIL",
        "accelerate",
        "bitsandbytes"
    ]
    
    missing_packages = []
    for package in required_packages:
        if not check_package(package):
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少以下包: {', '.join(missing_packages)}")
        print("建议安装命令:")
        print("pip install torch transformers datasets peft Pillow accelerate bitsandbytes")
    else:
        print("\n✓ 所有依赖包已安装")
        
        # 测试是否能导入 Qwen3VL 相关模块
        try:
            from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
            print("✓ 可以导入 Qwen3VL 模块")
            
            # 检查模型文件是否存在
            import os
            model_path = "./qwen3-vl-4b/unsloth/Qwen3-VL-4B-Instruct"
            if os.path.exists(model_path):
                print(f"✓ 模型文件存在: {model_path}")
                
                # 检查配置文件
                config_file = os.path.join(model_path, "config.json")
                if os.path.exists(config_file):
                    print(f"✓ 配置文件存在: {config_file}")
                else:
                    print(f"✗ 配置文件不存在: {config_file}")
            else:
                print(f"✗ 模型文件不存在: {model_path}")
                print("请先运行 download.py 下载模型")
                
        except ImportError as e:
            print(f"✗ 无法导入 Qwen3VL 模块: {e}")
            print("可能需要更新 transformers 库: pip install --upgrade transformers")
        except Exception as e:
            print(f"✗ 检查过程中出错: {e}")
    
    # 检查 CUDA 可用性
    print("\n检查 CUDA 可用性...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA 可用，设备: {torch.cuda.get_device_name(0)}")
            print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("✗ CUDA 不可用，将使用 CPU（速度会很慢）")
    except Exception as e:
        print(f"✗ 检查 CUDA 时出错: {e}")

if __name__ == "__main__":
    main()