import os
import urllib.request as request
import zipfile
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig
from pathlib import Path

class DataIngestion:
  def __init__(self,config:DataIngestionConfig):
    self.config = config
    
  def download_file(self):
    try:
      if not os.path.exists(self.config.local_data_file):
        filename,headers=request.urlretrieve(
          url=self.config.source_URL,
          filename=self.config.local_data_file
          )
        logger.info(f"{filename} downloaded! with following information:\n {headers}")
      else:
        logger.info(f"File already exists of size {get_size(Path(self.config.local_data_file))}")
    except Exception as e:
      logger.error(f"Error downloading {self.config.local_data_file}: {e}")
      # raise e
      
  def extract_zip_file(self):
    try:
      unzip_path = self.config.unzip_dir
      os.makedirs(unzip_path,exist_ok=True)
      with zipfile.ZipFile(self.config.local_data_file,'r') as zip_ref:
        zip_ref.extractall(path=unzip_path)
      logger.info(f"{self.config.local_data_file} extracted to {unzip_path}")
    except Exception as e:
      logger.error(f"Error extracting file {self.config.local_data_file} to {unzip_path}: {e}")