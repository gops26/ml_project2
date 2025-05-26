import sys
import os
# from src.logger import logging


def get_error_msgdetail(error, error_detail: sys):
    _,_,exe_tb = error_detail.exc_info()
    filename = exe_tb.tb_frame.f_code.co_filename
    error_message = f"error occured in line no  [{exe_tb.tb_lineno}], file [{filename}] error message : {str(error)} "
    return error_message


class CustomException(Exception):
    def __init__(self, errormsg: str, error_detail: sys):
        super().__init__(errormsg)
        self.error_message= get_error_msgdetail(errormsg, error_detail)

    def __str__(self):
        str(self.error_message)