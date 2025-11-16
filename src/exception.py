import sys
from src.logger import logging

class CustomException(Exception):
    """
    Custom excepion class that inherits from the base Exception class.
    """
    def __init__(self,error_message:str, error_detail:sys):
        super().__init__(error_message)
        self.error_message=self._generate_detailed_error_message(error_message,error_detail)
    
    @staticmethod
    def _generate_detailed_error_message(error_message: str, error_details:sys)->str:

        _,_,exc_tb=error_details.exc_info()
        file_name=exc_tb.tb_frame.f_code.co_filename

        detailed_message=(
            f"\n Error occured in Python script:"
            f"\n-<> File: {file_name}"
            f"\n-<> Line Number: {exc_tb.tb_lineno}"
            f"\n-<> Error Message: {str(error_message)}"
        )
    
        logging.error(detailed_message)
        return detailed_message

    def __str__(self)->str:
        """
        String representation of the exception.
        Returns:
            str: The detailed error message.
        """
        return self.error_message