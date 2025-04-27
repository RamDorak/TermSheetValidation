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

import pdfplumber
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

# Load sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_pdf_chunks(pdf_path):
    extracted_data = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text:
                continue  # Skip blank pages

            # Group text into paragraphs instead of single lines
            text_lines = text.split("\n")
            paragraphs = []
            current_para = []

            for line in text_lines:
                line = line.strip()
                if line:  # If the line is not empty, add to current paragraph
                    current_para.append(line)
                else:  # Empty line → new paragraph
                    if current_para:
                        paragraphs.append(" ".join(current_para))
                        current_para = []

            # Add remaining paragraph
            if current_para:
                paragraphs.append(" ".join(current_para))

            # Extract tables (unchanged)
            tables = page.extract_table()
            table_data = []
            if tables:
                df = pd.DataFrame(tables[1:], columns=tables[0])
                table_data = df.to_dict(orient="records")

            extracted_data.append({
                "page": page_num,
                "text_content": paragraphs,  # Store paragraphs instead of lines
                "tables": table_data
            })
    
    return extracted_data


def create_embedding_index(pdf_data):
    """
    Stores indexed paragraphs instead of lines and maintains previous/next context.
    """
    all_text = []
    index_to_page = {}

    for page in pdf_data:
        paragraphs = page["text_content"]  # Already grouped text
        for i, paragraph in enumerate(paragraphs):
            # Add paragraph
            all_text.append(paragraph)
            index_to_page[len(all_text) - 1] = page["page"]

            # Also store previous & next for expanded search
            prev_text = paragraphs[i - 1] if i > 0 else ""
            next_text = paragraphs[i + 1] if i < len(paragraphs) - 1 else ""
            all_text.append(f"{prev_text} {paragraph} {next_text}")
            index_to_page[len(all_text) - 1] = page["page"]

        # Index tables (unchanged)
        for table in page["tables"]:
            for key, value in table.items():
                entry = f"{key}: {value}"
                all_text.append(entry)
                index_to_page[len(all_text) - 1] = page["page"]
    
    # Convert to embeddings
    embeddings = model.encode(all_text, convert_to_numpy=True)
    dimension = embeddings.shape[1]

    # FAISS Indexing
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)

    return faiss_index, all_text, index_to_page

def get_validation_columns(validation_path):
    df = pd.read_excel(validation_path)
    return df.columns.tolist()

def find_relevant_chunks(column_names, faiss_index, all_text, index_to_page, top_k=10):
    relevant_data = {column: [] for column in column_names}

    for column in column_names:
        query_embedding = model.encode([column], convert_to_numpy=True)
        distances, indices = faiss_index.search(query_embedding, top_k)

        for idx in indices[0]:
            if idx == -1 or idx >= len(all_text):  # Handle invalid index
                continue
            
            if idx in index_to_page:
                page_num = index_to_page[idx]
                relevant_data[column].append(f"Page {page_num}: {all_text[idx]}")
    
    return relevant_data

def refine_with_genai(column_name, relevant_text, extracted_table):
    prompt = f"""
    You are an AI that extracts structured information from documents.
    Given the following extracted data, determine the most relevant information
    that should be entered under the column '{column_name}'. 
    I will also attach these table values for reference, if needed refer them for value too
    Table values '{extracted_table}'
    If you find nothing just return NA.
    
    Extracted Data:
    {json.dumps(relevant_text, indent=4)}
    
    Return only the most relevant value.
    """
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    
    return response.text.strip() if response.text else "No relevant data found"

def structure_data_with_genai(validation_columns, relevant_data, extracted_table):
    structured_data = {}
    
    for column, text_chunks in relevant_data.items():
        structured_data[column] = refine_with_genai(column, text_chunks, extracted_table)
        print(column," : ", structured_data[column])
        time.sleep(4)
        
    return structured_data

# File Paths
pdf_path = "ts.pdf"
validation_path = "rates.xlsx"

pdf_data = extract_pdf_chunks(pdf_path)

with open("chunks.json", "w") as f:
    json.dump(pdf_data, f, indent=4)

faiss_index, all_text, index_to_page = create_embedding_index(pdf_data)

validation_columns = get_validation_columns(validation_path)

validation_columns = ["Security Name", "Issuer", "IssueSizeand   Option   to retain over-subscription",
                    "ObjectsoftheIssue/ Details  of  the  utilization  of the proceeds", "Type of Instrument",
                    "Nature of Instrument", "Seniority", "Temporary  principal  write-down", "Listing",
                    "Tenor", "Convertibility", "Face Value", "CreditRating", "ModeofIssue", "Security",
                    "Coupon Rate", "Coupon Reset", "Coupon Type", "CouponPaymentFrequency", "CouponPaymentDates"]

relevant_data = find_relevant_chunks(validation_columns, faiss_index, all_text, index_to_page)

print("Relevant Data ",relevant_data)

extracted_table = [page["tables"] for page in pdf_data]

# Generate structured data
structured_data = structure_data_with_genai(validation_columns, relevant_data, extracted_table)

# Save Output
with open("structured_output.json", "w") as f:
    json.dump(structured_data, f, indent=4)

print("✅ PDF data extracted, structured, and validated successfully!")
