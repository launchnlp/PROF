import os
from typing import Union

def find_file_endswith(root: str, ext: str) -> Union[str, None]:
    '''
        Find the file with the given extension in the root directory
    '''

    for _, _, files in os.walk(root):
        for file in files:
            if file.endswith(ext):
                return os.path.join(root, file)
    return None