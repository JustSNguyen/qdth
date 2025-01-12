
import csv
import os
from models import TestDataRowInfo


def read_manual_labeled_files():
    relative_manual_label_csv_file_path = "../TestData/QSIDE Pharmacy Refusal Data Label - Web_based_2014-Present1-500byRelevence_cleaned.csv"

    current_directory = os.getcwd()
        
    full_manual_label_csv_file_path = os.path.join(current_directory, relative_manual_label_csv_file_path)

    manual_label_data_by_name = dict()

    with open(full_manual_label_csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        
        for row in csv_reader:
            document_name = row['Document name'].strip()
            
            about_pharmacy_refusals = row.get('about_pharmacy_refusals', '').strip().upper()
            additional_notes = row.get('Additional notes', '').strip()
            
            test_data_row_info = TestDataRowInfo(
                about_pharmacy_refusals=about_pharmacy_refusals, 
                additional_info=additional_notes
            )
            
            manual_label_data_by_name[document_name] = test_data_row_info
        
        return manual_label_data_by_name