# Module 4 Report  
### Frontend and Integration

## Introduction  
This module built simple HTML pages for interacting with all backend endpoints.

## High-Level Flow  
1. Text query page  
2. OCR upload page  
3. CNN image upload page  
4. Unified output formatting

## Description of Work  
- Built three HTML templates stored in templates/  
- Connected forms to Flask routes using POST  
- Displayed results using <pre> blocks for clarity  
- Verified that all three flows (text, OCR, CNN) return valid responses

## Key Decisions  
- Simple styling chosen for reliability  
- Kept output in raw JSON for transparency  
- Enabled file upload through multipart form

## Challenges and Solutions  
- JSON readability improved using pretty formatting  
- File upload issues fixed by enforcing correct MIME types  
- Layout kept minimal due to time limits

## Conclusion  
Module 4 joins all components together, producing a functional interface for query, OCR, and CNN recognition.

## References  
- HTML5 form upload patterns  
- Flask template engine  
