import dill
import os,sys
from src.logger import logging

def save_object(filepath:str, object:any):
    dill.dump(filepath, object)
    logging.info(f"saved object{object.__str__} to filepath {filepath} ") 