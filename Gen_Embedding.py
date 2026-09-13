from Data_Preprocessing import df
from sentence_transformers import SentenceTransformer

print('1. Downloading Embedding model>>>')
model = SentenceTransformer('all-MiniLM-L6-v2')

print('2. Generate Embeddings>>>')

embeddings = model.encode(df['cleaned_text'].tolist(), show_progress_bar = True)

df['Embedding'] = list(embeddings)

print('\n<<<< RESULT >>>>')
print(f"Summary count of generated embeddings: {len(df)}")
print(f"Size of the first vector: {len(df['Embedding'].iloc[0])}")
print('first 5 numbers of the first vector: ', df['Embedding'].iloc[0][:5])
