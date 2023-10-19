from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_training import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_evaluation import ModelEvaluationPipeline
# from cnnClassifier.pipeline.stage_03_prepare_callbacks import PrepareCallbacksTrainingPipeline

# STAGE_NAME="DATA INGESTION STAGE"

# try:
#   logger.info(f">>>>>> {STAGE_NAME} started. <<<<<<")
#   data_ingestion_training_pipeline = DataIngestionTrainingPipeline()
#   data_ingestion_training_pipeline.main()
#   logger.info(f">>>>>> {STAGE_NAME} completed successfully! <<<<<<\n\n")
# except Exception as e:
#   logger.info(f"{STAGE_NAME} failed: {e}")# logger.error(f"{STAGE_NAME} failed: {e}")# logger.exception(e)
#   raise e


# STAGE_NAME="PREPARE BASE MODEL STAGE"

# try:
#   logger.info(f">>>>>> {STAGE_NAME} started. <<<<<<")
#   prepare_base_model_training_pipeline = PrepareBaseModelTrainingPipeline()
#   prepare_base_model_training_pipeline.main()
#   logger.info(f">>>>>> {STAGE_NAME} completed successfully! <<<<<<\n\n")
# except Exception as e:
#   logger.info(f"{STAGE_NAME} failed: {e}")# logger.error(f"{STAGE_NAME} failed: {e}")# logger.exception(e)
#   raise e



# STAGE_NAME="Model Training STAGE"

# try:
#   logger.info(f">>>>>> {STAGE_NAME} started. <<<<<<")
#   model_training_pipeline = ModelTrainingPipeline()
#   model_training_pipeline.main()
#   logger.info(f">>>>>> {STAGE_NAME} completed successfully! <<<<<<\n\n")
# except Exception as e:
#   logger.info(f"{STAGE_NAME} failed: {e}")# logger.error(f"{STAGE_NAME} failed: {e}")# logger.exception(e)
#   raise e



STAGE_NAME="Model Evaluation STAGE"

try:
  logger.info(f">>>>>> {STAGE_NAME} started. <<<<<<")
  model_evaluation_pipeline = ModelEvaluationPipeline()
  model_evaluation_pipeline.main()
  logger.info(f">>>>>> {STAGE_NAME} completed successfully! <<<<<<\n\n")
except Exception as e:
  logger.info(f"{STAGE_NAME} failed: {e}")# logger.error(f"{STAGE_NAME} failed: {e}")# logger.exception(e)
  raise e



# STAGE_NAME="PREPARE CALLBACKS STAGE"

# try:
#   logger.info(f">>>>>> {STAGE_NAME} started. <<<<<<")
#   prepare_callbacks_training_pipeline = PrepareCallbacksTrainingPipeline()
#   prepare_callbacks_training_pipeline.main()
#   logger.info(f">>>>>> {STAGE_NAME} completed successfully! <<<<<<\n\n")
# except Exception as e:
#   logger.info(f"{STAGE_NAME} failed: {e}")# logger.error(f"{STAGE_NAME} failed: {e}")# logger.exception(e)
#   raise e
