from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_callbacks import PrepareCallbacks
from cnnClassifier import logger

STAGE_NAME="PREPARE CALLBACKS STAGE"

class PrepareCallbacksTrainingPipeline:
  
  def __init__(self) :
    pass
  
  def main(self):
    config_manager = ConfigurationManager()
    prepare_callbacks_config = config_manager.get_prepare_callbacks_config()
    prepare_callbacks = PrepareCallbacks(prepare_callbacks_config)
    prepare_callbacks.get_tb_ckpt_callbacks()
    
    
if __name__ == "__main__":
  try:
    logger.info(f"{STAGE_NAME} started.")
    obj = PrepareCallbacksTrainingPipeline()
    obj.main()
    logger.info(f"{STAGE_NAME} completed successfully!")
  except Exception as e:
    logger.info(f"{STAGE_NAME} failed: {e}")# logger.error(f"{STAGE_NAME} failed: {e}") # logger.exception(e)
    raise e