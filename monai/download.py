# download_mednist_monai.py
import os
from monai.apps import download_and_extract

def download_mednist_monai(data_dir="./MedNIST"):
    """使用MONAI下载MedNIST数据集"""
    os.makedirs(data_dir, exist_ok=True)
    
    url = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz"
    md5 = "0bc7306e7427e00ad1c5526a6677552d"
    
    print("Downloading MedNIST dataset using MONAI...")
    file_path = os.path.join(data_dir, "MedNIST.tar.gz")
    
    download_and_extract(
        url=url,
        filepath=file_path,
        output_dir=data_dir,
        hash_val=md5,
        hash_type="md5"
    )
    
    print("MedNIST dataset ready!")

if __name__ == "__main__":
    download_mednist_monai()
