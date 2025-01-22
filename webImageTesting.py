from duckduckgo_search import DDGS
from fastcore.all import L

def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    with DDGS() as ddgs:
        # Retrieve a list of search results with a specified maximum
        search_results = ddgs.images(keywords=term, max_results=max_images)
        # Extract the 'image' URLs from the search results
        image_urls = [result.get("image") for result in search_results]
        # Convert the list to a fastai L object (if needed)
        return L(image_urls)

# Example usage:
urls = search_images("dog images", max_images=10)
print(urls[0])