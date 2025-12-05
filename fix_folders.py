import os
import shutil

BASE = "scraped_images"

for filename in os.listdir(BASE):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    
    # example: 22112_CHOCOLATE_HOT_WATER_BOTTLE_0.jpg
    stock_code = filename.split("_")[0]

    folder_path = os.path.join(BASE, stock_code)
    os.makedirs(folder_path, exist_ok=True)

    src = os.path.join(BASE, filename)
    dst = os.path.join(folder_path, filename)

    shutil.move(src, dst)

print("✅ DONE — Images moved into folders by stock code")
