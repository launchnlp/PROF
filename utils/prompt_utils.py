from typing import List, Dict

example_sep = '[EXAMPLE_SEP]'
intra_example_sep = '[INTRA_EXAMPLE_SEP]'

def prompt_parser(prompt_file_text: str) -> List[Dict[str, str]]:
    '''
        Parses the prompt file into a message history
    '''
    prompt_message_list = prompt_file_text.split(example_sep)
    prompt_message_list_processed = filter(lambda x: x.strip() != '', map(lambda x: x.strip(), prompt_message_list))
    prompt = []
    for index, message in enumerate(prompt_message_list_processed):
        if index == 0:
            prompt.append({'role': 'system', 'content': message})
        elif message.strip() != '':
            user_message, assistant_message = message.split(intra_example_sep)
            prompt.append({'role': 'user', 'content': user_message})
            prompt.append({'role': 'assistant', 'content': assistant_message})
    return prompt