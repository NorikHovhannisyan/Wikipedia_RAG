import requests
from bs4 import BeautifulSoup
import pandas as pd


url = 'https://en.wikipedia.org/wiki/Python_(programming_language)'

headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(url, headers = headers)

soup = BeautifulSoup(response.text, 'html.parser')

parags = soup.find_all('p')

scraped_data = []

for p in parags:
    text = p.text.strip()
    if len(text) > 50:
        scraped_data.append(text)

print(f"Successfully scraped {len(scraped_data)} from Wikipedia:\n")

for i, p_text in enumerate(scraped_data[:4], 1):
    print(f"{i}. {p_text}\n")