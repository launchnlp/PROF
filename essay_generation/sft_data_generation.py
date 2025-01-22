import os
import json
import argparse

def generate_sft_data(
    data_dir: str,
    num_samples: int,
    output_file: str,
    start_index: int = 0
):
    '''
        To process the essays in an sft format
        with input = title and output = essay
    '''
    data_list = []
    
    for essay_index in range(num_samples):

        # reading the essay file
        essay_file = os.path.join(data_dir, 'essay{}.txt'.format(str(essay_index + start_index + 1).zfill(3)))
        with open(essay_file, 'r') as f:
            essay = f.read()
        
        # first line is title and the rest is essay
        title, essay_str = essay.split('\n', 1)
        data_list.append({
            'instruction': title.strip(),
            'input': '',
            'output': essay_str.strip()
        })

    # writing the data to the output file as json
    with open(output_file, 'w') as f:
        json.dump(data_list, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script for extracting the data from the Essays dataset and framing it as a title to text generation')
    parser.add_argument('--data_dir', type=str, default='/home/inair/data/ArgumentAnnotatedEssays-2.0/brat-project-final')
    parser.add_argument('--num_samples', type=int, default=350)
    parser.add_argument('--output_file', type=str, default='/home/inair/data/essay_writing_350.json')
    parser.add_argument('--start_index', type=int, default=0)
    args = parser.parse_args()

    generate_sft_data(
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        output_file=args.output_file,
        start_index=args.start_index
    )
