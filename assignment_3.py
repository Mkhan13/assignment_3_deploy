import requests
import os
from flask import Flask, request, jsonify
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dotenv import load_dotenv
import xml.etree.ElementTree as ET


load_dotenv() # Load environment variables from .env file

app = Flask(__name__) # Create Flask application

embedding_model = SentenceTransformer("all-MiniLM-L6-v2") # Load pre-trained sentence embedding model

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY")) # Initialize Pinecone vector database with API key
index_name = "researchpapers"

if index_name not in [i.name for i in pc.list_indexes()]: # Ensure the index is created before accessing it
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2") # Initialize Hugging Face GPT-2 model
tokenizer.pad_token = tokenizer.eos_token  # Padding token

def fetch_research_papers(query):
    '''
    Function to fetch and preprocess research papers from arXiv API
    '''

    base_url = "http://export.arxiv.org/api/query"
    params = {"search_query": f"all:{query}", "start": 0, "max_results": 50}
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        papers = response.text
        vectors = []

        root = ET.fromstring(papers) # Parse XML response from arXiv API

        # Iterate through each research paper entry in the response
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title = entry.find("{http://www.w3.org/2005/Atom}title").text # Extract the title of the paper
            abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text # Extract the abstract/summary
            url = entry.find("{http://www.w3.org/2005/Atom}id").text # Extract the papers URL

            embedding = embedding_model.encode(abstract).tolist() # Generate embedding vector for abstract
            paper_id = url.split("/")[-1]  # Extract unique paper ID

            vectors.append((paper_id, embedding, {"title": title, "abstract": abstract, "url": url})) # Append the ID, embedding, and metadata to the vector array

        index.upsert(vectors) # Store research paper vectors in Pinecone
        return {"message": "Research Papers Loaded into Pinecone Vector DB"}
    else:
        return {"error": "Failed to fetch data", "details": response.text}

# API Endpoint to fetch and store research papers in Pinecone
@app.route("/load_papers", methods=["GET"])
def load_papers():
    query = request.args.get("query")
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    return jsonify(fetch_research_papers(query))

def retrieve_papers(query):
    '''
    Function to retrieve relevant papers and generate AI-based response using GPT-2
    '''
    query_embedding = embedding_model.encode(query).tolist()
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

    matches = results.get("matches", [])
    if not matches:
        return {"response": "No relevant research papers found."}

    # Construct retrieved context from matching research papers
    retrieved_context = []

    for item in matches:
        title = item['metadata']['title']
        abstract = item['metadata']['abstract'][:500]  # Limit abstract to 500 characters because I am limited on characters
        url = item['metadata']['url']
        
        formatted_entry = f"Title: {title}, Abstract: {abstract}, URL: {url}"
        retrieved_context.append(formatted_entry)

    retrieved_context = " ".join(retrieved_context)  # Join all the formatted entries into a string

    # Input prompt for GPT-2 Hugging Face model
    prompt = f"""
    You are an AI research assistant specialized in retrieving and summarizing academic papers.
    Given a user query, provide relevant research insights based on indexed papers. Ensure responses are detailed, well-structured, and cite relevant papers.

    Context: {retrieved_context}
    User Query: {query}
    Answer:
    """
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1000, padding=True) # Tokenize the input, limit max_length to account for extra characters
    outputs = model.generate( # Generate a response using GPT-2
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_p=0.9,
        temperature=0.7,
        max_new_tokens=150,
        do_sample=True
    )

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True) # Decode model output to generate readable text response
    return {"Response": response_text}

# API Endpoint to retrieve research papers and generate AI response
@app.route("/search", methods=["GET"])
def search_papers():
    query = request.args.get("query")
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    return jsonify(retrieve_papers(query))


# Run the Flask app
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)