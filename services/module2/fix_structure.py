import os
import shutil

src = "scraped_images"

for img_file in os.listdir(src):
    if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    # Example filename: 22112_CHOCOLATE_HOT_WATER_BOTTLE_0.jpg
    parts = img_file.split("_")
    stock_code = parts[0]  # First part MUST be stock code

    stock_folder = os.path.join(src, stock_code)
    os.makedirs(stock_folder, exist_ok=True)

    src_path = os.path.join(src, img_file)
    dst_path = os.path.join(stock_folder, img_file)

    shutil.move(src_path, dst_path)

print("✅ Folder restructuring complete!")
