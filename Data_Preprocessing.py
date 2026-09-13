import pandas as pd
import re
import requests
from bs4 import BeautifulSoup

url = 'https://en.wikipedia.org/wiki/Python_(programming_language)'
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers = headers)
soup = BeautifulSoup(response.text, 'html.parser')

paragraphs = soup.find_all('p')
raw_data = [p.text.strip() for p in paragraphs if len(p.text.strip()) > 50]

df = pd.DataFrame(raw_data, columns = ['raw_text'])

def clean_text(text):
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

df['cleaned_text'] = df['raw_text'].apply(clean_text)
df['char_count'] = df['cleaned_text'].apply(len)

print('\n<<< Example of cleaned data>>>')
print('First text (clean): ')
print(df['cleaned_text'].iloc[0])

print('\n2nd text (clean): ')
print(df['cleaned_text'].iloc[1])

print('\n3rd text (clean): ')
print(df['cleaned_text'].iloc[2])

print('\n4th text (clean): ')
print(df['cleaned_text'].iloc[3])