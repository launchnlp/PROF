import openai
import time
import asyncio
from typing import List, Dict
from openai import AsyncOpenAI, OpenAI

'''
    Defining a normal client
'''
openai_client = OpenAI()

'''
    Defining a async client
'''
async_openai_client = AsyncOpenAI()

def openai_chat_api(
    messages: List[Dict[str, str]],
    engine: str = 'gpt-35-turbo',
    temperature: float = 0.0,
    max_tokens: int = 2000,
    top_p: float = 0.95,
    frequency_penalty: float = 0,
    presence_penalty: float = 0,
    stop: List[str] = None,
    num_retries: int = 5
):
    '''
        Calls open ai chat api
    '''

    for _ in range(num_retries):
        try:
            response = openai_client.chat.completions.create(
                model=engine,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop
            )
            return response
        except Exception as e:
            print(e)
            print('Retrying call to openai chat api')
            time.sleep(5)

    return None

async def async_openai_chat_api(
    messages: List[Dict[str, str]],
    engine: str = 'gpt-35-turbo',
    temperature: float = 0.0,
    max_tokens: int = 2000,
    top_p: float = 0.95,
    frequency_penalty: float = 0,
    presence_penalty: float = 0,
    stop: List[str] = None,
    num_retries: int = 5
):
    '''
        Calls open ai chat api
    '''

    for _ in range(num_retries):
        try:
            response = await async_openai_client.chat.completions.create(
                model=engine,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop
            )
            return response
        except Exception as e:
            print(e)
            print('Retrying call to openai chat api')
            await asyncio.sleep(5)

    return None

async def openai_chat_api_batch(
    messages_batch: List[List[Dict[str, str]]],
    **kwargs,
):
    '''
        Calls open ai chat api in batch
    '''

    async def process_messages(messages):  
        return await async_openai_chat_api(messages, **kwargs) 
    
    tasks = []  
    for messages in messages_batch:  
        task = asyncio.ensure_future(process_messages(messages))  
        tasks.append(task)  
  
    responses = await asyncio.gather(*tasks)  
    return responses

if __name__ == '__main__':
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "What is the purpose of life?"
        }
    ]
    import time

    messages_batch = [messages for _ in range(5)]

    # running it in synchronous manner
    tic = time.time()
    response_list = []
    for messages in messages_batch:
        response = openai_chat_api(messages, temperature=1.0)
        response_list.append(response)
    toc = time.time()
    print('Time elapsed for synchronous call: {}'.format(toc - tic))

    # running it in asynchronous manner
    tic = time.time()
    response_list = asyncio.run(openai_chat_api_batch(messages_batch, temperature=1.0))
    toc = time.time()
    print('Time elapsed for asynchronous call: {}'.format(toc - tic))

    import pdb; pdb.set_trace()
