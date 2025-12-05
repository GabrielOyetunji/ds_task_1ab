import os
import re
import time
from typing import Dict, List
import requests
from bs4 import BeautifulSoup

# Fixed labels for the image dataset
CATEGORIES: Dict[str, str] = {
    "bag": "fashion lunch bag product photo white background",
    "clock": "alarm clock home decor product photo white background",
    "mug": "ceramic mug cup set product photo white background",
    "candle": "candle holder tealight candle product photo",
    "basket": "wicker storage basket product photo",
    "toy": "toy figurine kids toy white background",
    "kitchenware": "kitchenware utensils plates bowls product photo",
    "lantern": "metal lantern candle holder product photo"
}

class ImageScraper:
    """
    Scrapes images for predefined categories and saves them into folders.
    Used to build a dataset for image classification.
    """

    def __init__(self,
                 save_root: str = "cnn_data",
                 images_per_class: int = 50,
                 pause_seconds: float = 0.6):

        self.save_root = save_root
        self.images_per_class = images_per_class
        self.pause_seconds = pause_seconds

        self.raw_dir = os.path.join(self.save_root, "raw")
        os.makedirs(self.raw_dir, exist_ok=True)

    def _safe_filename(self, text: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
        return cleaned[:80]

    def _download_image(self, url: str, path: str) -> bool:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            with open(path, "wb") as f:
                f.write(response.content)

            return True

        except Exception as e:
            print("  Skipped image:", e)
            return False

    def scrape_category(self, label: str, query: str) -> List[str]:
        print(f"Scraping: {label}")

        output_dir = os.path.join(self.raw_dir, label)
        os.makedirs(output_dir, exist_ok=True)

        search_url = "https://www.bing.com/images/search"
        params = {"q": query.replace(" ", "+")}
        headers = {"User-Agent": "Mozilla/5.0"}

        try:
            response = requests.get(search_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
        except Exception as e:
            print("Error loading page:", e)
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        images = soup.select("img.mimg")

        saved_paths = []
        count = 0

        for idx, img in enumerate(images):
            if count >= self.images_per_class:
                break

            src = img.get("src") or img.get("data-src")
            if not src:
                continue

            filename = f"{label}_{idx}.jpg"
            dest_path = os.path.join(output_dir, filename)

            if self._download_image(src, dest_path):
                saved_paths.append(dest_path)
                count += 1

            time.sleep(self.pause_seconds)

        print(f"{label}: {len(saved_paths)} images saved.")
        return saved_paths

    def run(self):
        results = {}

        for label, query in CATEGORIES.items():
            results[label] = self.scrape_category(label, query)

        print("\nScraping completed.")
        return results


if __name__ == "__main__":
    scraper = ImageScraper(
        save_root="cnn_data",
        images_per_class=50,
        pause_seconds=0.6
    )
    scraper.run()
    print("Done.")