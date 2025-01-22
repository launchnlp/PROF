import os
import re
import argparse
from utils.doc_utils import *
from tqdm import tqdm

def process_submissions_wrapper(
    input_folder: str,
    output_folder: str
) -> None:
    
    # check if the input_folder is a directory
    if not os.path.isdir(input_folder):
        raise ValueError(f"Input folder {input_folder} is not a directory")
    
    # create output folder
    os.makedirs(output_folder, exist_ok=True)

    # iterate over the files in the input folder
    for root, dirs, files in os.walk(input_folder):
        for file in tqdm(files):
            
            # custom processing for docx
            if file.endswith('.docx'):
                doc_text = read_docx(os.path.join(root, file))
                doc_text = doc_cleaner(doc_text)
            elif file.endswith('.pdf'):
                doc_text = extract_text_from_pdf_with_formatting(os.path.join(root, file))
            doc_text = doc_cleaner(doc_text)

            # save the processed text
            if file.endswith('.docx') or file.endswith('.pdf'):

                # extracting the document id
                student_id = re.findall(r'_(\d+)_', file)[0]

                # save the processed text
                with open(os.path.join(output_folder, f'{student_id}.txt'), 'w') as f:
                    f.write(doc_text)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Process the submissions from the mwrite data')
    parser.add_argument('--input_folder', type=str, help='The folder containing the submissions', default='/home/inair/data/econ_data/2203106_First Draft - Writing Assignment 2: Government Intervention/submissions')
    parser.add_argument('--output_folder', type=str, help='The folder to save the processed submissions', default='/home/inair/data/econ_data/assignment_2_processed/revised_essay')
    args = parser.parse_args()

    process_submissions_wrapper(args.input_folder, args.output_folder)