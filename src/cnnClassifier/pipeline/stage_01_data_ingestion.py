from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier import logger

STAGE_NAME="DATA INGESTION STAGE"

class DataIngestionTrainingPipeline:
  
  def __init__(self) :
    pass
  
  def main(self):
    config_manager = ConfigurationManager()
    data_ingestion_config = config_manager.get_data_ingestion_config()
    data_ingestion = DataIngestion(data_ingestion_config)
    data_ingestion.download_file()
    data_ingestion.extract_zip_file()
    
    
if __name__ == "__main__":
  try:
    logger.info(f"{STAGE_NAME} started.")
    data_ingestion_training_pipeline = DataIngestionTrainingPipeline()
    data_ingestion_training_pipeline.main()
    logger.info(f"{STAGE_NAME} completed successfully!")
  except Exception as e:
    logger.info(f"{STAGE_NAME} failed: {e}")# logger.error(f"{STAGE_NAME} failed: {e}") # logger.exception(e)
    raise e