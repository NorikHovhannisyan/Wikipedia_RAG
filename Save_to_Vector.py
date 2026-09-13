import chromadb
from Gen_Embedding import df
from sentence_transformers import SentenceTransformer

print("1. Starting ChromaDB initialization...")
chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="python_wikipedia_knowledge")
print("2. Adding Data and vectors into Vector DB...")

# Ֆիլտրում ենք շատ կարճ տեքստերը (օրինակ՝ 100 նիշից պակաս), որ որոնման որակը բարձրանա
filtered_df = df[df["cleaned_text"].str.len() > 100].copy()

documents = filtered_df["cleaned_text"].tolist()
embeddings = filtered_df["Embedding"].tolist()  # Վերցնում ենք քո գեներացրած վեկտորները
ids = [f"doc_{i}" for i in range(len(filtered_df))]

# Ավելացնում ենք և՛ տեքստերը, և՛ քո պատրաստի վեկտորները (embeddings)
collection.add(documents=documents, embeddings=embeddings, ids=ids)

print("\n<<< Successfully saved >>>")
print(f"Count of saved documents in Vector DB: {collection.count()}")

test_query = "Who is the author, creator or founder of Python?"
print(f"\nTest question to Base: '{test_query}'")

# 1. Հարցումը սարքում ենք վեկտոր նույն մոդելով
model = SentenceTransformer("all-MiniLM-L6-v2")
query_embedding = model.encode(test_query).tolist()

# 2. Որոնում ենք բազայում՝ օգտագործելով query_embeddings (այլ ոչ թե query_texts)
results = collection.query(query_embeddings=[query_embedding], n_results=3)

print("\n<<< Most similar text found from base >>>")
for i, doc in enumerate(results["documents"][0], 1):
  print(f"\n[{i}] {doc}")