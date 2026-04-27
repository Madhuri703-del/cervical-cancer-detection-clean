import os
import shutil

source_base = "."
target_base = "original_dataset"

classes = {
    "im_Dyskeratotic": "Dyskeratotic",
    "im_Koilocytotic": "Koilocytotic",
    "im_Metaplastic": "Metaplastic",
    "im_Parabasal": "Parabasal",
    "im_Superficial-Intermediate": "Superficial"
}

for src_folder, dest_folder in classes.items():
    src_path = os.path.join(source_base, src_folder, src_folder, "CROPPED")  # ✅ FIX

    dest_path = os.path.join(target_base, dest_folder)
    os.makedirs(dest_path, exist_ok=True)

    count = 0
    for file in os.listdir(src_path):
        if file.lower().endswith(".bmp"):
            shutil.copy(os.path.join(src_path, file), os.path.join(dest_path, file))
            count += 1

    print(f"Copied {count} images to {dest_folder}")

print("✅ Done")