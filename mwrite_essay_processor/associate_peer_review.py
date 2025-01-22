import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from utils.mwrite_utils import find_file_endswith, format_rubric_data

def associate_peer_review_wrapper(
    first_version_dir: str,
    first_draft_dir: str,
    output_file: str
) -> None:
    '''
        Associate the scores with the processed submissions
    '''

    # check if the first_version_dir is a directory
    if not os.path.isdir(first_version_dir):
        raise ValueError(f"First version directory {first_version_dir} is not a directory")
    
    # check if the first_draft_dir is a directory
    if not os.path.isdir(first_draft_dir):
        raise ValueError(f"First draft directory {first_draft_dir} is not a directory")
    
    # create output folder
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # creating the final data dictionary
    final_data = {}

    # read the rubrics file
    rubrics_file = find_file_endswith(first_draft_dir, '_rubrics.csv')
    if not rubrics_file:
        raise ValueError('Rubrics file not found in the first draft directory')
    rubrics_df = pd.read_csv(rubrics_file)
    
    # read the grades file
    grades_file = find_file_endswith(first_draft_dir, '_grades.csv')
    if not grades_file:
        raise ValueError('Grades file not found in the first draft directory')
    grades_df = pd.read_csv(grades_file)

    # iterating over the grades file
    for _, row in tqdm(grades_df.iterrows()):
        
        # parsing the rubric data
        peer_review_list = format_rubric_data(row['rubric_data'], rubrics_df)

        # check if the row['submitter_id'] is already present in the data_dict
        if row['submitter_id'] in final_data:
            final_data[row['submitter_id']]['feedback'][row['grader_id']] = peer_review_list
        else:
            
            # check if the row['submitter_id'] is present in the first_version_dir
            if os.path.exists(os.path.join(first_version_dir, f'{row["submitter_id"]}.txt')):
                with open(os.path.join(first_version_dir, f'{row["submitter_id"]}.txt'), 'r') as f:
                    essay = f.read()

                # add the essay to the final_data
                final_data[row['submitter_id']] = {
                    'essay': essay,
                    'feedback': {
                        row['grader_id']: peer_review_list
                    }
                }

    # save the final data
    with open(output_file, 'w') as f:
        json.dump(final_data, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Associate the scores with the processed submissions')
    parser.add_argument('--first_version_dir', type=str, help='The folder where first version of the essay is stored', default='/home/inair/data/econ_data/assignment_2_processed/essay')
    parser.add_argument('--first_draft_dir', type=str, help='The folder where all the information related to the first draft is stored', default='/home/inair/data/econ_data/2203106_First Draft - Writing Assignment 2: Government Intervention')
    parser.add_argument('--output_file', type=str, help='The final output file', default='/home/inair/data/econ_data/assignment_2_processed/parsed.json')
    args = parser.parse_args()

    associate_peer_review_wrapper(args.first_version_dir, args.first_draft_dir, args.output_file)