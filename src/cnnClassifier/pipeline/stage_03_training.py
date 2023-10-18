from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_callbacks import PrepareCallbacks
from cnnClassifier.components.training  import Training
from cnnClassifier import logger

STAGE_NAME="Model Training Stage"

class ModelTrainingPipeline:
  
  def __init__(self) :
    pass
  
  def main(self):
    config_manager = ConfigurationManager()
    
    prepare_callbacks_config = config_manager.get_prepare_callbacks_config()
    prepare_callbacks = PrepareCallbacks(prepare_callbacks_config)
    callback_list=prepare_callbacks.get_tb_ckpt_callbacks() 
    
    training_config = config_manager.get_training_config()
    training = Training(training_config)
    training.get_base_model()
    training.train_valid_generator()
    training.train(
      callback_list=callback_list
    )
    
    
if __name__ == "__main__":
  try:
    logger.info(f">>>>>> {STAGE_NAME} started. <<<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> {STAGE_NAME} completed successfully! <<<<<<")
  except Exception as e:
    logger.info(f"{STAGE_NAME} failed: {e}")# logger.error(f"{STAGE_NAME} failed: {e}") # logger.exception(e)
    raise e
  