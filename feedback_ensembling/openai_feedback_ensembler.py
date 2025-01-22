from typing import List, Dict
from utils.openai_utils import openai_chat_api, openai_chat_api_batch

def ensemble_feedback(
    prompt: List[Dict[str, str]],
    title: str,
    essay: str,
    feedback_list: List[str],
    chat_model: str = 'gpt-35-turbo',
    temperature: float = 0.0,
) -> str:
    '''
        Applies feedback to the input essay along with the feedback
    '''

    # creating the full message
    message = prompt + [
        {
            'role': 'user',
            'content': 'Essay: {}\n{}\n\nFeedback: {}'.format(title, essay, '\n--x--\n'.join(feedback_list))
        }
    ]

    # calling the openai chat api
    response = openai_chat_api(message, chat_model, temperature=temperature)
    try:
        return response.choices[0].message.content.strip()
    except:
        return essay
    
def feedback_selector(
    prompt: List[Dict[str, str]],
    title: str,
    essay: str,
    feedback_list: List[str],
    chat_model: str = 'gpt-35-turbo',
    temperature: float = 0.0,
) -> str:
    '''
        Applies feedback to the input essay along with the feedback
    '''

    # creating the content message
    content = 'Essay: {}\n{}\n\nFeedback:\n'.format(title, essay)
    for index, feedback in enumerate(feedback_list):
        content += '[{}]\n{}\n'.format(index + 1, feedback)

    # creating the full message
    message = prompt + [
        {
            'role': 'user',
            'content': content
        }
    ]

    # calling the openai chat api
    response = openai_chat_api(message, chat_model, temperature=temperature)
    try:
        return response.choices[0].message.content.strip()
    except:
        return essay