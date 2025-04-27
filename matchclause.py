import google.generativeai as genai
from fuzzywuzzy import fuzz
import json
from tabulate import tabulate

# Initialize Gemini API
genai.configure(api_key="AIzaSyCrg-AR84IqDwYWgeED5yG2Gh7hpVwZfn0")

# Load Gemini model
model = genai.GenerativeModel("gemini-2.0-flash")

# Hardcoded structured term sheet
term_sheet = {
    "Interest Rate": "7.25",
    "Loan Amount": "5000000",
    "Maturity Date": "December 31, 2030",
    "Covenant Clause": "The borrower must maintain a minimum DSCR of 1.25x."
}

# Hardcoded validation sheet
validation_sheet = {
    "Interest Rate": "7.20",
    "Loan Amount": "5,000,000",
    "Maturity Date": "31-12-2030",
    "Covenant Clause": "The borrower is required to maintain a Debt Service Coverage Ratio (DSCR) of at least 1.25."
}

# Function to compare numeric values using fuzzy matching
def compare_numeric(value1, value2):
    return fuzz.ratio(value1, value2)

# Function to assess text values using Gemini API and return JSON
def assess_text(term_text, validation_text, field_name):
    prompt = f"""
    Compare these two text clauses and return a valid JSON output in this exact format:
    
    {{
        "Field": "{field_name}",
        "Confidence Score": <integer from 0-100>,
        "Discrepancy Level": "<High Risk Discrepancy | Medium Discrepancy | No Discrepancy>"
    }}

    Term Sheet Clause: "{term_text}"
    Validation Sheet Clause: "{validation_text}"
    """

    response = model.generate_content(prompt)

    # Print raw response for debugging
    print(f"Raw Response for {field_name}: {response.text}")

    try:
        # Extract JSON from response safely
        parsed_response = json.loads(response.text.strip("```json").strip("```"))
        print("Parsed Response ", parsed_response)
        return parsed_response
    except json.JSONDecodeError:
        return {"Field": field_name, "Confidence Score": 0, "Discrepancy Level": "Error Parsing Response"}

# Processing the term sheet data
results = []
for key, term_value in term_sheet.items():
    if key in validation_sheet:
        validation_value = validation_sheet[key]
        
        if term_value.replace('.', '', 1).isdigit():  # Numeric comparison
            score = compare_numeric(term_value, validation_value)
            results.append({
                "Field": key,
                "Term Sheet Value": term_value,
                "Validation Sheet Value": validation_value,
                "Confidence Score": score,
                "Discrepancy Level": "Low" if score > 80 else "Medium" if score > 50 else "High"
            })
        else:  # Text comparison using Gemini API
            analysis = assess_text(term_value, validation_value, key)
            results.append(analysis)

# Print results in formatted JSON
print(json.dumps(results, indent=4))