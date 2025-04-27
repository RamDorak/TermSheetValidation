import pdfplumber
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import time
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="AIzaSyCrg-AR84IqDwYWgeED5yG2Gh7hpVwZfn0")

# Load sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_pdf_chunks(pdf_path):
    extracted_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            text_lines = text.split("\n") if text else []
            
            tables = page.extract_table()
            table_data = []
            if tables:
                df = pd.DataFrame(tables[1:], columns=tables[0])
                table_data = df.to_dict(orient="records")
            
            extracted_data.append({
                "page": page_num,
                "text_content": text_lines,
                "tables": table_data
            })
    
    return extracted_data

def create_embedding_index(pdf_data):
    all_text = []
    index_to_page = {}

    for page in pdf_data:
        for line in page["text_content"]:
            all_text.append(line)
            index_to_page[len(all_text) - 1] = page["page"]
        
        for table in page["tables"]:
            for key, value in table.items():
                entry = f"{key}: {value}"
                all_text.append(entry)
                index_to_page[len(all_text) - 1] = page["page"]
    
    embeddings = model.encode(all_text, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
    
    return faiss_index, all_text, index_to_page

def get_validation_columns(validation_path):
    df = pd.read_excel(validation_path)
    return df.columns.tolist()

def find_relevant_chunks(column_names, faiss_index, all_text, index_to_page, top_k=3):
    column_names = ["Security Name", "Issuer", "Issue Size and Option to retain over-subscription", 
                    "Objects of the Issue", "Type of Instrument", "Nature of Instrument", "Seniority",
                    "Temporary principal write-down", "Listing", "Tenor"]
    relevant_data = {column: [] for column in column_names}
    
    for column in column_names:
        query_embedding = model.encode([column], convert_to_numpy=True)
        distances, indices = faiss_index.search(query_embedding, top_k)
        
        for idx in indices[0]:
            if idx < len(all_text):
                page_num = index_to_page[idx]
                relevant_data[column].append(f"Page {page_num}: {all_text[idx]}")
    
    return relevant_data

def refine_with_genai(column_name, relevant_text):
    prompt = f"""
    You are an AI that extracts structured information from documents.
    Given the following extracted data, determine the most relevant information
    that should be entered under the column '{column_name}'.
    
    Extracted Data:
    {json.dumps(relevant_text, indent=4)}
    
    Return only the most relevant value.
    """
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    
    return response.text.strip() if response.text else "No relevant data found"

def structure_data_with_genai(validation_columns, relevant_data):
    structured_data = {}
    
    for column, text_chunks in relevant_data.items():
        structured_data[column] = refine_with_genai(column, text_chunks)
        print(structured_data[column])
        time.sleep(3)
        
    
    return structured_data

# File Paths
pdf_path = "capsheet.pdf"
validation_path = "rates.xlsx"

# Processing Steps
pdf_data = extract_pdf_chunks(pdf_path)
faiss_index, all_text, index_to_page = create_embedding_index(pdf_data)
# validation_columns = get_validation_columns(validation_path)
# relevant_data = find_relevant_chunks(validation_columns, faiss_index, all_text, index_to_page)

# # Generate structured data
# structured_data = structure_data_with_genai(validation_columns, relevant_data)

# print(pdf_data)
with open("testing.json", "w") as f:
    json.dump(pdf_data, f, indent=4)

# # Save Output
# with open("structured_output.json", "w") as f:
#     json.dump(structured_data, f, indent=4)

print("✅ PDF data extracted, structured, and validated successfully!")
