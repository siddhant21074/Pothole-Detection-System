import sys
from pathlib import Path

# Add project root to Python path - handle both direct execution and module import
if __file__:
    project_root = Path(__file__).resolve().parent.parent
else:
    # Fallback if __file__ is not available
    project_root = Path.cwd()

# Ensure project root is in Python path
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.logger import logging

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename 
    error_message = 'Error Occured in python script name [{0}] line numner [{1}] error message [{2}]'.format(
        file_name,exc_tb.tb_lineno,str(error)) 
    return error_message  
    
class CustomException(Exception):
    def __init__(self, error_message,error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_details)
        
    def __str__(self):
        return self.error_message

if __name__ == "__main__":
    obj = CustomException()
    obj.__init__()