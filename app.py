from flask import Flask, request, jsonify, render_template
from services.module1.recommendation_service import RecommendationService
from services.module2.ocr_service import OCRService
from services.module3.detection_service import DetectionService

app = Flask(__name__)

# Initialize all service classes
recommendation_service = RecommendationService()
ocr_service = OCRService()
detection_service = DetectionService()


@app.route("/")
def home():
    # Simple home route listing available endpoints
    return jsonify({
        "message": "E-commerce Product API",
        "endpoints": {
            "/recommend": "POST - Text query → product recommendations",
            "/ocr-recommend": "POST - Image → OCR → recommendations",
            "/detect-product": "POST - Product image → CNN classification + recommendations",
            "/query-page": "Frontend page for text query",
            "/ocr-page": "Frontend page for OCR query",
            "/detect-page": "Frontend page for product detection"
        }
    })


# Show page for direct text queries
@app.route("/query-page")
def query_page():
    return render_template("query_page.html")


# Show page for uploading OCR images
@app.route("/ocr-page")
def ocr_page():
    return render_template("ocr_page.html", data=None)


# Show page for uploading a product image for CNN detection
@app.route("/detect-page")
def detect_page():
    return render_template("detect_page.html")


# Text query → recommendations
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()

    if not data or "query" not in data:
        return jsonify({"success": False, "error": "Missing query"}), 400

    query = data["query"]
    top_k = data.get("top_k", 5)

    result = recommendation_service.get_recommendations(query, top_k=top_k)

    return jsonify(result), (200 if result["success"] else 400)


# OCR → text extraction → recommendations
@app.route("/ocr-recommend", methods=["POST"])
def ocr_recommend():
    # Check if an image was uploaded
    if "image" not in request.files:
        return render_template("ocr_page.html", data={"error": "No image uploaded"})

    # Save the uploaded image
    image = request.files["image"]
    image_path = "uploaded_input.jpg"
    image.save(image_path)

    # Extract text using OCR
    extracted_text = ocr_service.extract_text(image_path)

    if not extracted_text:
        return render_template("ocr_page.html", data={"error": "Could not extract text"})

    # Get recommendations using extracted text
    result = recommendation_service.get_recommendations(extracted_text)
    result["extracted_text"] = extracted_text

    # Render updated OCR page with results
    return render_template("ocr_page.html", data=result)


# CNN detection → predicted class → recommendations
@app.route("/detect-product", methods=["POST"])
def detect_product():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    # Save uploaded product image
    image = request.files["image"]
    img_path = "uploaded_product.jpg"
    image.save(img_path)

    # Predict class using the CNN model
    predicted_class = detection_service.predict(img_path)

    # Get recommendations using the predicted class
    result = recommendation_service.get_recommendations(predicted_class)
    result["predicted_class"] = predicted_class

    return jsonify(result), 200


# Run Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)