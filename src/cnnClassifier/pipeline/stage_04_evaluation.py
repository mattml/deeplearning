from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_callbacks import PrepareCallbacks
from cnnClassifier.components.evaluation import Evaluation
from cnnClassifier import logger

STAGE_NAME="Model Evaluation Stage"

class ModelEvaluationPipeline:
  
  def __init__(self) :
    pass
  
  def main(self):
    
    config_manmager=ConfigurationManager()
    evaluation_config=config_manmager.get_evaluation_config()
    evaluation=Evaluation(evaluation_config)
    # evaluation.load_model()
    evaluation.evaluation() #加载模型，创建数据生成器，评估，保存分数
    # evaluation.save_model_score()
    evaluation.log_into_mlflow() # 评估日志发送到mlflow平台
    
   
    
    
    
if __name__ == "__main__":
  try:
    logger.info(f">>>>>> {STAGE_NAME} started. <<<<<<")
    obj = ModelEvaluationPipeline()
    obj.main()
    logger.info(f">>>>>> {STAGE_NAME} completed successfully! <<<<<<")
  except Exception as e:
    logger.info(f"{STAGE_NAME} failed: {e}")# logger.error(f"{STAGE_NAME} failed: {e}") # logger.exception(e)
    raise e