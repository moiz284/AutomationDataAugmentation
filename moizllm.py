import os
import gdown
import pandas as pd
import csv
import json
import re
import time
import google.generativeai as genai

# Download the file from Google Drive using its shareable link

input_folder = "moiz"  # Folder containing input CSV files
output_folder = "output_folder"
# os.makedirs(output_folder, exist_ok=True)
# Configure Google GenAI with API key
api_key = os.getenv("GEMINI_API_KEY")  # Ensure this environment variable is set
if not api_key:
    raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")
genai.configure(api_key=api_key)

model_name = "gemini-1.5-flash"

# Load the dataset
# file_path = 'moiz/split_pak_file_3.csv'  # Adjust file path for GitHub Actions
base_prompt = """
Provide the following JSON for each unique job role. Ensure no duplicate rows in the output. Each row should represent a distinct job role, identified by a unique combination of attributes like "Job Title", "Company", "Date Posted".

[
    {
        "Job Title": "",
        "Company": "",
        "Location": "",
        "Date Posted": "",
        "Salary": "",
        "Technical skills": "",
        "Industry": "",
        "Experience required": "",
        "Soft skills": "",
        "Education needed": "",
        "Personality traits": {
            "MBTI": "Determine the MBTI type based on the job title and job description. Use the dominant cognitive functions and behavioral patterns that align with the role. Format: 'MBTI Type', e.g., 'INTJ' for a strategic planner role.",
            "RIASEC": "Comma-separated dominant traits (e.g., 'Investigative, Conventional'). Derive from job responsibilities and nature of work.",
            "Big Five": {
                "Conscientiousness": "Rate on a scale of 1-5 based on job demands (e.g., highly structured roles like 'Accountant' would be rated 5, while creative roles like 'Graphic Designer' may be rated lower).",
                "Openness": "Assess based on the role's need for creativity and adaptability. High for research and creative roles, lower for process-driven jobs.",
                "Extraversion": "Rate high for customer-facing or leadership roles, lower for solitary analytical work.",
                "Agreeableness": "High for roles requiring teamwork, empathy, or customer service; lower for competitive, high-pressure environments.",
                "Neuroticism": "Consider job stress levels; fast-paced, high-risk roles may have higher neuroticism scores."
            }
        }
    }
]

Guidelines:
1. Use double quotes for all keys/values to ensure valid JSON.
2. *MBTI Assessment:* Identify the most relevant MBTI type by analyzing the job title and description. Consider cognitive functions, decision-making style, and interaction patterns. For example:
   - A "Data Scientist" might be *INTP (Logician)* due to analytical and problem-solving skills.
   - A "Marketing Manager" might be *ENTP (Debater)* for strategic thinking and persuasion.
3. *Big Five Traits:* Evaluate based on job description:
   - *Conscientiousness* → Higher for structured, detail-oriented roles (e.g., Accountants, Project Managers).
   - *Openness* → Higher for creative, exploratory roles (e.g., Designers, Researchers).
   - *Extraversion* → Higher for leadership and public-facing roles (e.g., Sales, Management).
   - *Agreeableness* → Higher for teamwork-oriented roles (e.g., HR, Social Work).
   - *Neuroticism* → Higher for stressful, high-stakes roles (e.g., Emergency Responders).
4. *Industry Classification:* Assign the correct *"Industry"* based on the *nature of the job itself, using standardized categories like **"Information Technology," "Healthcare," "Finance," "Legal Services,"* etc. Do *not* determine the industry based on the company's domain—focus on the actual job role and responsibilities.
   - An *"Office Boy"* should fall under *"Administrative & Support Services,"* regardless of whether they work in a tech company, hospital, or bank.
   - A *"Legal Advisor"* should be categorized under *"Legal Services,"* even if they work at a financial institution.
5. *RIASEC Traits:* Assign based on the nature of work. For example:
   - *Software Engineers* → Investigative, Conventional
   - *Graphic Designers* → Artistic, Investigative
   - *Teachers* → Social, Enterprising
6. *Experience Formatting:* Standardize experience requirements using *only numbers and "months"/"years"*, following these rules:
   - If a job states a *specific number* (e.g., "7+ years required"), use *only the number and unit* → "7 years".
   - If experience is *preferred*, set it to → "0 years".
   - If no experience requirement is mentioned, set it to → "0 years".
   - If experience is mentioned but *not quantified* (e.g., "Hands-on experience required"), assume a default → "6 months".
7. *Education Formatting:* Standardize degree requirements using these rules:
   - If a specific field is mentioned → "Bachelors in Computer Science", "Masters in Psychology", etc.
   - If only degree level is mentioned → "Bachelors Any", "Masters Any".
   - For lower levels → "Matric", "Inter".
   - If no education requirement is provided → "N/A".
8. Extract "Technical skills" and "Soft skills" directly from the job description, ensuring alignment with the role’s responsibilities.
9. Standardize job titles where possible (e.g., "Software Engineer" and "Software Developer" should have a unified classification).
10. If any information (e.g., salary, education required) is missing or not specified, mark it as "N/A."
11. Ensure the final output is a valid JSON array with no syntax errors.
""" # Base prompt content remains unchanged
sub_batch_size=2
max_retries = 5
rate_limit_delay = 5

output_file = os.path.join(output_folder, "output1.csv")
skipped_file = os.path.join(output_folder, "skipped_batches.txt")
error_log_file = os.path.join(output_folder, "error_logs.txt")
output_folder = "output_folder"  # Set a GitHub-compatible output folder

# Prepare output files
for file in [output_file, skipped_file, error_log_file]:
    with open(file, mode="w", encoding="utf-8") as f:
        f.write("")  # Clear content
for i in range(3, 3 * 17, 3):  # Assuming files are named split_pak_file_3.csv, split_pak_file_6.csv, etc.
    file_name = f"split_pak_file_{i}.csv"
    file_path = os.path.join(input_folder, file_name)

    if not os.path.exists(file_path):
        print(f"File {file_name} not found. Skipping...")
        continue

    print(f"Processing file: {file_name}")
    try:
        df = pd.read_csv(file_path)
        print(f"File {file_name} loaded successfully.")
        print(df.head())  # Display the first few rows for confirmation
    except Exception as e:
        print(f"Error loading file {file_name}: {e}")
        continue

    # Process the dataset in sub-batches
    for sub_batch_start in range(0, len(df), sub_batch_size):
        sub_batch = df.iloc[sub_batch_start:sub_batch_start + sub_batch_size]
        data_excerpt = sub_batch.to_string(index=False)
        prompt = f"Here is a dataset:\n{data_excerpt}\n{base_prompt}"
        print(f"Processing sub-batch starting at index {sub_batch_start}")

        model = genai.GenerativeModel(model_name)
        retry_count = 0
        response_content = None

        # Retry logic for Generative AI requests
        while retry_count < max_retries:
            try:
                response = model.generate_content(prompt)
                response_content = response.text.strip()
                time.sleep(rate_limit_delay)  # Avoid rate-limiting
                break
            except Exception as e:
                retry_count += 1
                wait_time = rate_limit_delay * retry_count
                print(f"Request failed: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        if response_content is None:
            print(f"Max retries reached for sub-batch {sub_batch_start}. Skipping...")
            with open(skipped_file, mode="a", encoding="utf-8") as log_file:
                log_file.write(f"Skipped sub-batch {sub_batch_start}:\n{data_excerpt}\n")
            continue

        # Parse response and write to output
        try:
            response_content = re.search(r"\[.*?\]", response_content, re.DOTALL).group(0)
            data = json.loads(response_content)
        except Exception as e:
            print(f"Error decoding JSON for sub-batch {sub_batch_start}: {e}")
            with open(error_log_file, mode="a", encoding="utf-8") as log_file:
                log_file.write(f"Error for sub-batch {sub_batch_start}:\n{response_content}\n")
            continue

        # Validate and write JSON response to CSV
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            with open(output_file, mode="a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                if csvfile.tell() == 0:  # Write header only for the first time
                    writer.writerow(data[0].keys())
                for item in data:
                    writer.writerow(item.values())
            print(f"Sub-batch data successfully written to {output_file}.")
        else:
            print(f"The response for sub-batch {sub_batch_start} is not in the expected JSON list format.")
            with open(error_log_file, mode="a", encoding="utf-8") as log_file:
                log_file.write(f"Invalid format for sub-batch {sub_batch_start}:\n{response_content}\n")
