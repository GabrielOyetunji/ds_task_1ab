"""
Product Recommendation Service

Provides natural language product recommendations using vector similarity search.
"""

import logging
from typing import Dict, List, Optional
from services.module1.vector_service import VectorService


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RecommendationService:
    """Handles product recommendations based on natural language queries."""
    
    def __init__(self) -> None:
        """Initialize recommendation service with vector search."""
        try:
            logger.info("Initializing RecommendationService...")
            self.vector_service = VectorService()
            self.vector_service.create_index(delete_if_exists=False)
            logger.info("RecommendationService ready")
        except Exception as e:
            logger.error(f"Failed to initialize RecommendationService: {e}")
            raise

    def validate_query(self, query: str) -> bool:
        """Validate user query for safety and format.
        
        Args:
            query: User's search query
            
        Returns:
            True if query is valid, False otherwise
        """
        if not query or not isinstance(query, str):
            return False
        
        query = query.strip()
        
        if len(query) < 2:
            logger.warning("Query too short")
            return False
        
        if len(query) > 200:
            logger.warning("Query too long")
            return False
        
        return True

    def generate_natural_language_response(
        self,
        query: str,
        products: List[Dict]
    ) -> str:
        """Generate natural language response from search results.
        
        Args:
            query: Original user query
            products: List of matching products
            
        Returns:
            Natural language description of results
        """
        if not products:
            return f"I couldn't find any products matching '{query}'. Try a different search term."
        
        response = f"Based on your search for '{query}', I found {len(products)} matching product(s):\n\n"
        
        for i, product in enumerate(products, 1):
            response += (
                f"{i}. {product['description']} "
                f"(£{product['price']:.2f}) - "
                f"{product['similarity_score']*100:.1f}% match\n"
            )
        
        return response

    def get_recommendations(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict:
        """Get product recommendations for a query.
        
        Args:
            query: Natural language search query
            top_k: Number of recommendations to return
            
        Returns:
            Dictionary with success status, products, and natural language response
        """
        try:
            # Validate query
            if not self.validate_query(query):
                logger.warning(f"Invalid query rejected: {query}")
                return {
                    "success": False,
                    "error": "Invalid query. Must be 2-200 characters.",
                    "query": query,
                    "products": [],
                    "response": "Please provide a valid search query (2-200 characters)."
                }
            
            logger.info(f"Processing recommendation request: '{query}'")
            
            # Search for similar products
            products = self.vector_service.search_similar_products(query, top_k)
            
            # Generate natural language response
            nl_response = self.generate_natural_language_response(query, products)
            
            logger.info(f"Returning {len(products)} recommendations")
            
            return {
                "success": True,
                "query": query,
                "products": products,
                "response": nl_response
            }
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "products": [],
                "response": "An error occurred while searching. Please try again."
            }


if __name__ == "__main__":
    try:
        service = RecommendationService()
        result = service.get_recommendations("red heart decoration")
        print(result['response'])
    except Exception as e:
        logger.error(f"Test failed: {e}")
