import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from Gen_Embedding import df
from dotenv import load_dotenv

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name = 'python_knowledge_rag')

filtered_df = df[df['cleaned_text'].str.len() > 100].copy()
documents = filtered_df['cleaned_text'].tolist()
embeddings = filtered_df['Embedding'].tolist()
ids = [f"doc_{i}" for i in range(len(filtered_df))]

collection.add(documents = documents, embeddings = embeddings, ids = ids)
model = SentenceTransformer('all-MiniLM-L6-v2')
load_dotenv()
api_key = os.getenv('GROQ_API_KEY')
groq_client = Groq(api_key = api_key)

def ask_rag(user_question):
    print(f"\n==========================================================")
    print(f"User's question: '{user_question}'")
    
    query_embedding = model.encode(user_question).tolist()
    
    results = collection.query(query_embeddings = [query_embedding], n_results = 3)
    retrieved_docs = results['documents'][0]
    context = '\n\n'.join(retrieved_docs)
    
    system_prompt = f"""
    Դու խելացի օգնական ես: Պատասխանիր օգտատիրոջ հարցին՝ օգտվելով ՄԻԱՅՆ ստորև բերված տեքստային տվյալներից (Context):
    Պատասխանիր հայերենով, հստակ և կարճ:
    Եթե տվյալների մեջ պատասխանը չկա, ասա. «Տրամադրված տվյալներում պատասխանը առկա չէ»:

    CONTEXT:
    {context}
    """
    
    chat_completion = groq_client.chat.completions.create(
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': (
              f"Translate this question to English briefly for search:"
              f" '{user_question}'"
          )},
        ],
        model = 'openai/gpt-oss-safeguard-20b',
    )
    
    print('\n<<<< Final Answer Of AI (GROQ RAG) >>>>')    
    print(chat_completion.choices[0].message.content)
    
ask_rag('Who created the Python language and when?')
