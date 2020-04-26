from pathlib import Path
from glob import glob

def get_basename(file_path):
    return file_path.split('/')[-1].split('.')[0]

def get_image_files(path):
    p = Path(path)
    files = []
    for ext in ('*.jpg','*.png'):
        files.extend([str(item) for item in p.rglob(ext)])
    return files

if __name__ == "__main__":
    for item in get_image_files("/data/ocr/handwrite/real_data"):
        print(item)
        print(get_basename(item))