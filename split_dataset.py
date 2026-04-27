import os
import shutil
import random

source = "original_dataset"
target = "dataset"

train_ratio = 0.7
val_ratio = 0.15

for cls in os.listdir(source):
    cls_path = os.path.join(source, cls)
    images = os.listdir(cls_path)
    random.shuffle(images)

    train_end = int(len(images) * train_ratio)
    val_end = int(len(images) * (train_ratio + val_ratio))

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    for split, data in zip(["train","val","test"], [train_imgs,val_imgs,test_imgs]):
        path = os.path.join(target, split, cls)
        os.makedirs(path, exist_ok=True)

        for img in data:
            shutil.copy(os.path.join(cls_path, img), os.path.join(path, img))

print("✅ Dataset Split Done!")