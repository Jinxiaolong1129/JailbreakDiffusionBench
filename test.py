from google.cloud import language_v1
from concurrent.futures import ThreadPoolExecutor
import time

class ContentModerator:
    """
    A class to handle both single and batch content moderation
    """
    def __init__(self):
        self.client = language_v1.LanguageServiceClient()
    
    def moderate_text(self, text_content):
        """
        Moderate a single text using Google's Text Moderation API
        
        Args:
          text_content: The text content to analyze.
          
        Returns:
          The moderation response containing safety attributes and confidence scores.
        """
        # Set up the document with the content to moderate
        document = {
            "content": text_content,
            "type": language_v1.Document.Type.PLAIN_TEXT,
            "language": "en"  # Optional: specify language or let API auto-detect
        }
        
        # Call the moderateText method
        response = self.client.moderate_text(document=document)
        
        return response
    
    def moderate_texts(self, texts, max_workers=20):
        """
        Moderate multiple texts in parallel
        
        Args:
          texts: A single text string or a list of text strings to moderate
          max_workers: Maximum number of parallel workers for batch processing
          
        Returns:
          A single response or list of responses in the same order as input
        """
        # Handle single text case
        if isinstance(texts, str):
            return self.moderate_text(texts)
        
        # Handle list of texts (batch processing)
        results = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_text = {executor.submit(self.moderate_text, text): i for i, text in enumerate(texts)}
            
            # Collect results as they complete
            for future in future_to_text:
                try:
                    result = future.result()
                    # Store result with original index to maintain order
                    results.append((future_to_text[future], result))
                except Exception as exc:
                    print(f'Text generated an exception: {exc}')
                    results.append((future_to_text[future], None))
        
        # Sort by original index and return just the results
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

def print_moderation_results(response):
    """
    Print the moderation results in a readable format
    """
    print("Moderation Results:")
    print("-----------------")
    
    for category in response.moderation_categories:
        print(f"{category.name}: {category.confidence:.2f}")

def print_batch_results(responses, texts):
    """
    Print batch moderation results in a readable format
    
    Args:
      responses: List of moderation responses
      texts: List of original texts
    """
    for i, (response, text) in enumerate(zip(responses, texts)):
        print(f"\nText {i+1}: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
        if response:
            print_moderation_results(response)
        else:
            print("Failed to get moderation results for this text")

# Example usage
if __name__ == "__main__":
    # Example texts to moderate
    texts_to_moderate = [
        "Hello, how are you doing today?",
        "Fuck your mother.",
        "I hate everything about you and wish you would die.",
        "The weather is quite nice today.",
        "This product is absolutely terrible, I want a refund."
    ]*100
    
    # Initialize the moderator
    moderator = ContentModerator()
    
    # Example 1: Single text moderation
    print("EXAMPLE 1: SINGLE TEXT MODERATION")
    single_text = "This is a test message."
    single_response = moderator.moderate_texts(single_text)
    print_moderation_results(single_response)
    
    # Example 2: Batch text moderation
    print("\nEXAMPLE 2: BATCH TEXT MODERATION")
    # Measure performance
    start_time = time.time()
    
    # Get batch moderation results - simply pass the list directly
    batch_responses = moderator.moderate_texts(texts_to_moderate)
    
    end_time = time.time()
    
    # Print results
    print_batch_results(batch_responses, texts_to_moderate)
    
    # Print performance metrics
    print(f"\nBatch processing completed in {end_time - start_time:.2f} seconds")
    print(f"Average time per text: {(end_time - start_time) / len(texts_to_moderate):.2f} seconds")
    
    # Example of how to check if any content exceeds specific thresholds
    for i, response in enumerate(batch_responses):
        if response:
            for category in response.moderation_categories:
                if category.name == "Toxic" and category.confidence > 0.7:
                    print(f"Warning: Text {i+1} detected as {category.name} with high confidence")