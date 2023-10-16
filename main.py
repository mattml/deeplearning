from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME="DATA INGESTION STAGE"

try:
  logger.info(f"{STAGE_NAME} started.")
  data_ingestion_training_pipeline = DataIngestionTrainingPipeline()
  data_ingestion_training_pipeline.main()
  logger.info(f"{STAGE_NAME} completed successfully!")
except Exception as e:
  logger.info(f"{STAGE_NAME} failed: {e}")# logger.error(f"{STAGE_NAME} failed: {e}")# logger.exception(e)
  raise e
