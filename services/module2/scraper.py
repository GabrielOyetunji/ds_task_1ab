"""
Image Scraping Service for CNN Training Data

Scrapes product images from web sources to build training dataset for CNN model.
"""

import os
import re
import time
import logging
from typing import Dict, List
import requests
from bs4 import BeautifulSoup


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
    """Scrapes product images for CNN training dataset.
    
    Downloads images from web search results and organizes them
    by category for model training.
    """

    def __init__(
        self,
        save_root: str = "cnn_data",
        images_per_class: int = 50,
        pause_seconds: float = 0.6
    ) -> None:
        """Initialize image scraper.
        
        Args:
            save_root: Root directory to save scraped images
            images_per_class: Number of images to download per category
            pause_seconds: Delay between downloads to avoid rate limiting
        """
        self.save_root = save_root
        self.images_per_class = images_per_class
        self.pause_seconds = pause_seconds

        self.raw_dir = os.path.join(self.save_root, "raw")
        os.makedirs(self.raw_dir, exist_ok=True)
        
        logger.info(f"ImageScraper initialized: {images_per_class} images/class")
        logger.info(f"Output directory: {self.raw_dir}")

    def _safe_filename(self, text: str) -> str:
        """Convert text to safe filename.
        
        Args:
            text: Original text
            
        Returns:
            Sanitized filename
        """
        cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
        return cleaned[:80]

    def _download_image(self, url: str, path: str) -> bool:
        """Download image from URL to local path.
        
        Args:
            url: Image URL
            path: Local path to save image
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            with open(path, "wb") as f:
                f.write(response.content)

            return True

        except Exception as e:
            logger.debug(f"Skipped image: {e}")
            return False

    def scrape_category(self, label: str, query: str) -> List[str]:
        """Scrape images for a specific product category.
        
        Args:
            label: Category label (e.g., 'bag', 'clock')
            query: Search query string
            
        Returns:
            List of paths to downloaded images
        """
        logger.info(f"Scraping category: {label}")

        output_dir = os.path.join(self.raw_dir, label)
        os.makedirs(output_dir, exist_ok=True)

        search_url = "https://www.bing.com/images/search"
        params = {"q": query.replace(" ", "+")}
        headers = {"User-Agent": "Mozilla/5.0"}

        try:
            response = requests.get(search_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Error loading search page for {label}: {e}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        images = soup.select("img.mimg")
        
        logger.info(f"Found {len(images)} potential images for {label}")

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
                
                if count % 10 == 0:
                    logger.info(f"{label}: {count}/{self.images_per_class} downloaded")

            time.sleep(self.pause_seconds)

        logger.info(f"{label}: {len(saved_paths)} images saved to {output_dir}")
        return saved_paths

    def run(self) -> Dict[str, List[str]]:
        """Execute scraping for all categories.
        
        Returns:
            Dictionary mapping category labels to lists of image paths
        """
        logger.info("="*50)
        logger.info("STARTING IMAGE SCRAPING")
        logger.info(f"Categories: {list(CATEGORIES.keys())}")
        logger.info("="*50)
        
        results = {}

        for label, query in CATEGORIES.items():
            results[label] = self.scrape_category(label, query)

        total_images = sum(len(paths) for paths in results.values())
        
        logger.info("="*50)
        logger.info("SCRAPING COMPLETED")
        logger.info(f"Total images downloaded: {total_images}")
        logger.info(f"Categories completed: {len(results)}")
        logger.info("="*50)
        
        return results


if __name__ == "__main__":
    try:
        scraper = ImageScraper(
            save_root="cnn_data",
            images_per_class=50,
            pause_seconds=0.6
        )
        results = scraper.run()
        logger.info("Task 5 completed successfully")
    except Exception as e:
        logger.error(f"Task 5 failed: {e}")
        raise
