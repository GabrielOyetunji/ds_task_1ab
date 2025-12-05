import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from services.module1.vector_service import VectorService


class RecommendationService:
    def __init__(self):
        self.vector_service = VectorService()
        self.vector_service.create_index(delete_if_exists=False)

    def validate_query(self, query):
        if not query or not isinstance(query, str):
            return False, None, "Query must be a non-empty string"

        query = query.strip()

        if len(query) < 2:
            return False, None, "Query is too short. Please provide at least 2 characters"

        if len(query) > 200:
            return False, None, "Query is too long. Please limit to 200 characters"

        return True, query, None

    def generate_natural_language_response(self, query, products):
        if not products:
            return f"I couldn’t find any items related to '{query}'. Try using a clearer description."

        best = products[0]
        response = (
            f"Here are some items related to '{query}'. "
            f"The closest match is {best['description']} (£{best['price']:.2f}). "
        )

        if len(products) > 1:
            response += "You may also consider: "
            for p in products[1:4]:
                response += f"{p['description']} (£{p['price']:.2f}), "
            response = response.rstrip(", ") + "."

        return response

    def get_recommendations(self, query, top_k=5):
        is_valid, cleaned_query, error_message = self.validate_query(query)

        if not is_valid:
            return {
                "success": False,
                "error": error_message,
                "products": [],
                "response": error_message
            }

        try:
            products = self.vector_service.search_similar_products(
                cleaned_query, top_k=top_k
            )

            natural_text = self.generate_natural_language_response(
                cleaned_query, products
            )

            return {
                "success": True,
                "query": cleaned_query,
                "products": products,
                "response": natural_text
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Search failed: {str(e)}",
                "products": [],
                "response": "An unexpected error occurred. Please try again."
            }


if __name__ == "__main__":
    service = RecommendationService()
    sample_query = "Looking for a red heart decoration"
    result = service.get_recommendations(sample_query)

    print(result["response"])
    for p in result["products"]:
        print("-", p["description"], p["similarity_score"])