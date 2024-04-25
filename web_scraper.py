import requests
from bs4 import BeautifulSoup

def fetch_images(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    images = [(img['src'], img.get('alt', 'No description')) for img in soup.find_all('img')]
    return images
