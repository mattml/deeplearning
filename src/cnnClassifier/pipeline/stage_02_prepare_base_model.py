from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_base_model import PrepareBasemModel
from cnnClassifier import logger

STAGE_NAME="PREPARE BASE MODEL STAGE"

class PrepareBaseModelTrainingPipeline:
  
  def __init__(self) :
    pass
  
  def main(self):
    config_manager = ConfigurationManager()
    prepare_base_model_config = config_manager.get_prepare_base_model_config()
    prepare_base_model = PrepareBasemModel(prepare_base_model_config)
    prepare_base_model.get_base_model()
    prepare_base_model.update_base_model()
    
    
if __name__ == "__main__":
  try:
    logger.info(f"{STAGE_NAME} started.")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f"{STAGE_NAME} completed successfully!")
  except Exception as e:
    logger.info(f"{STAGE_NAME} failed: {e}")# logger.error(f"{STAGE_NAME} failed: {e}") # logger.exception(e)
    raise e