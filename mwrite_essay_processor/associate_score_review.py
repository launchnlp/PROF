import os
import json
import argparse
import pandas as pd
from typing import Dict, Any
from utils.mwrite_utils import find_file_endswith

def associate_score_review_wrapper(
    peer_review_draft_dir: str,
    parsed_file: str
) -> None:
    '''
        Associates the score for each reviewer used for training the reward model
    '''
    # check if the peer_review_draft_dir is a directory
    if not os.path.isdir(peer_review_draft_dir):
        raise ValueError(f"Peer review draft directory {peer_review_draft_dir} is not a directory")
    
    # check if the parsed_file is a file
    if not os.path.isfile(parsed_file):
        raise ValueError(f"Parsed file {parsed_file} is not a file")
    with open(parsed_file, 'r') as file:
        parsed_data: Dict[str, Any] = json.load(file)

    # reading the _grades.csv
    grades_file = find_file_endswith(peer_review_draft_dir, '_grades.csv')
    if not grades_file:
        raise ValueError('Grades file not found in the peer review draft directory')
    grades_df = pd.read_csv(grades_file)
    
    # iterate through each data element in the parsed_data
    for submitter_id in parsed_data.keys():

        # creating a dictionary to store the scores of each reviewer
        parsed_data[submitter_id]['feedback_score'] = {}
        
        # iterating over the ids of each reviewer
        for reviewer_id in parsed_data[submitter_id]['feedback'].keys():
            
            # getting row from the grade file with submitter id == reviewer id
            reviewer_row = grades_df[grades_df['submitter_id'] == int(reviewer_id)]
            
            # check if the reviewer_row is empty
            if reviewer_row.empty:
                continue
            reviewer_score = float(reviewer_row['score'].iloc[0])
            parsed_data[submitter_id]['feedback_score'][reviewer_id] = reviewer_score

    # write the parsed_data to the parsed_file
    with open(parsed_file, 'w') as file:
        json.dump(parsed_data, file, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Associate score for each reviewer used for training reward model')
    parser.add_argument('--peer_review_draft_dir', type=str, help='Directory containing the peer review scores. I am using the draft terminology to keep the naming consistent with other other scripts', default='/home/inair/data/econ_data/2203110_Peer Review - Writing Assignment 2: Government Intervention')
    parser.add_argument('--parsed_file', type=str, help='The file generated through the first stage of the pipeline', default='/home/inair/data/econ_data/assignment_2_processed/parsed.json')
    args = parser.parse_args()

    # call the associate_score_review_wrapper
    associate_score_review_wrapper(
        args.peer_review_draft_dir,
        args.parsed_file
    )