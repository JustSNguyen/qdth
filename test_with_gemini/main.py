import csv
import os
import random 

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic import BaseModel
from dotenv import load_dotenv

from data import read_manual_labeled_files
from prompts import classify_chat_prompt
from models import ClassificationResponse

import time 

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

gemini_api_key = os.getenv('GEMINI_API_KEY')
def get_predicted_label_for_file(file_path):
    content = read_file(file_path)
    prompt = classify_chat_prompt.format(source=content)
    model = GeminiModel('gemini-1.5-flash', api_key=gemini_api_key)
    agent = Agent(model, result_type=ClassificationResponse, system_prompt=prompt)
    result = agent.run_sync(prompt)
    return result.data 

if __name__ == "__main__":
    load_dotenv()

    manual_label_data_by_name = read_manual_labeled_files()

    folder_paths = ["../FullData/Web_based_2014-Present1-500byRelevence_cleaned"]

    result_csv_file_name = "validation_result.csv"

    processed = 0 
    start_time = time.time()
    with open(result_csv_file_name, 'w', newline='', encoding='utf-8') as result_csv_file:
        result_csv_file_writer = csv.writer(result_csv_file)
        
        result_csv_file_writer.writerow(['document_name', 'actual_label', 'gemini_model_predicted_label', 'additional_info'])

        for folder_path in folder_paths:
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)

                filename_without_extension = os.path.splitext(filename)[0].strip()
            
                # Some file names contain a strange leading apostrophe ('), which for some reasons is omitted when copy to CSV file so this is a "hack" to handle those documents 
                if filename_without_extension[0] == "'":
                    filename_without_extension = filename_without_extension[1:].strip()

                actual_data = manual_label_data_by_name[filename_without_extension]
                if actual_data.not_labeled():
                    continue 

                actual_label = actual_data.about_pharmacy_refusals
                try:
                    gemini_response = get_predicted_label_for_file(file_path)
                    about_pharmaceutical_refusals = gemini_response.about_pharmaceutical_refusals
                    additional_information = gemini_response.additional_information
                    result_csv_file_writer.writerow([filename_without_extension, actual_label, about_pharmaceutical_refusals, additional_information])

                    processed += 1 
                    # clear the screen before printing the next file
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print(f"Processed {processed} files")
                    
                    # Since Gemini(free tier) has a rate limit of 15 requests per minutes, we need to sleep for 60/15 = 4 seconds between each request
                    time.sleep(4)

                except Exception as e:
                    print(f"Error: {e}")
        
        end_time = time.time()
        duration_in_seconds = end_time - start_time
        duration_in_minutes = duration_in_seconds / 60
        print(f"Processed {processed} files in {duration_in_minutes} minutes")