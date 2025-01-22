import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from utils.mwrite_utils import find_file_endswith, format_rubric_data

def associate_revised_essay_wrapper(
    revised_version_dir: str,
    revised_draft_dir: str,
    parsed_file: str
) -> None:
    
    # check if the revised_version_dir is a directory
    if not os.path.isdir(revised_version_dir):
        raise ValueError(f"Revised version directory {revised_version_dir} is not a directory")
    
    # check if the revised_draft_dir is a directory
    if not os.path.isdir(revised_draft_dir):
        raise ValueError(f"Revised draft directory {revised_draft_dir} is not a directory")
    
    # read the parsed file
    if not os.path.isfile(parsed_file):
        raise ValueError(f"Parsed file {parsed_file} is not a file")
    with open(parsed_file, 'r') as file:
        parsed_data = json.load(file)

    # reading the _grades.csv
    grades_file = find_file_endswith(revised_draft_dir, '_grades.csv')
    if not grades_file:
        raise ValueError('Grades file not found in the revised draft directory')
    grades_df = pd.read_csv(grades_file)
    
    # reading the _rubrics.csv
    rubrics_file = find_file_endswith(revised_draft_dir, '_rubrics.csv')
    if not rubrics_file:
        raise ValueError('Rubrics file not found in the revised draft directory')
    rubrics_df = pd.read_csv(rubrics_file)
    
    # iterate through each data element in the parsed_data
    for submitter_id in tqdm(parsed_data.keys()):
        
        # read the essay file in the revised_version_dir
        if os.path.exists(os.path.join(revised_version_dir, f'{submitter_id}.txt')):
            with open(os.path.join(revised_version_dir, f'{submitter_id}.txt'), 'r') as f:
                revised_essay = f.read()

            # add the revised_essay to the parsed_data
            parsed_data[submitter_id]['revised_essay'] = revised_essay

        else:
            continue

        # check if the submitter_id is present in the grades_df
        if int(submitter_id) not in grades_df['submitter_id'].values:
            parsed_data[submitter_id]['revised_overall_score'] = None
            parsed_data[submitter_id]['revised_score'] = None

        # get the row in the grades_df with the submitter_id
        else:
            row = grades_df[grades_df['submitter_id'] == int(submitter_id)].iloc[0]
            parsed_data[submitter_id]['revised_overall_score'] = float(row['score'])

            # parsing the rubric data
            revised_score = format_rubric_data(row['rubric_data'], rubrics_df, value='points')
            revised_score_obj = {}
            for score_obj in revised_score:
                revised_score_obj[score_obj['description']] = {'score': score_obj['comment']}
            parsed_data[submitter_id]['revised_score'] = revised_score_obj

    # save the parsed_data
    with open(parsed_file, 'w') as file:
        json.dump(parsed_data, file, indent=4)




if __name__ == '__main__':
    parser = argparse.ArgumentParser('Associate the revised essay with the processed submission')
    parser.add_argument('--revised_version_dir', type=str, help='The folder where the revised version of the essay is stored as a text file', default='/home/inair/data/econ_data/assignment_2_processed/revised_essay')
    parser.add_argument('--revised_draft_dir', type=str, help='The folder where the revised draft of the essay is stored in the mwrite format', default='/home/inair/data/econ_data/2203117_Revised Draft - Writing Assignment 2: Government Interventio')
    parser.add_argument('--parsed_file', type=str, help='The file generated through the first stage of the pipeline', default='/home/inair/data/econ_data/assignment_2_processed/parsed.json')
    args = parser.parse_args()

    associate_revised_essay_wrapper(
        args.revised_version_dir,
        args.revised_draft_dir,
        args.parsed_file
    )
