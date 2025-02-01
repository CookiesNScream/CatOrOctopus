from duckduckgo_search import DDGS
from fastcore.all import L
import fastai
from fastbook import *
from fastai.vision.widgets import *

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
urls = search_images("octopus in the wild", max_images=200)
dest = 'imageDest'
print(urls[3])
download_url(urls[0], dest)
sample_img = PILImage.create(dest)
sample_img.to_thumb(128, 128)