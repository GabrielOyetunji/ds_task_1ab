# Module 2 Report  
### OCR and Web Scraping

## Introduction  
This module introduced OCR capability and automated image scraping needed for CNN training.

## High-Level Flow  
1. OCR for extracting handwritten or printed text  
2. Use text for product recommendation  
3. Scrape product images for CNN training  
4. Organize scraped images by class

## Description of Work  
- Integrated Tesseract for OCR  
- Implemented preprocess steps and text extraction  
- Built scraping tool that downloads images for each stock code  
- Folder structure cleanup ensured correct CNN image loading

## Key Decisions  
- Selected Tesseract because it works reliably offline  
- Chose three image downloads per class to match dataset limits  
- Added fallback checks when OCR returns empty strings

## Challenges and Solutions  
- Poor lighting in scanned images required grayscale and thresholding  
- OCR accuracy varied; normalization improved results  
- Scraper blocking resolved by rotating the user agent

## Conclusion  
Module 2 established OCR and produced a usable image dataset for CNN training.

## References  
- Pytesseract  
- Requests and BeautifulSoup  
- OpenCV
