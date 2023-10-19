import os
import zipfile
import tensorflow as tf
import urllib.request as request
import time
from cnnClassifier.config.configuration import EvaluationConfig
from pathlib import Path
from cnnClassifier.utils.common import save_json
import mlflow
import dagshub
from urllib.parse import urlparse


class Evaluation:
  def __init__(self,config=EvaluationConfig):
    self.config = config
    
  def _valid_generator(self):
    
    datagenerator_kwargs = dict(
        rescale = 1./255,# validation_split=0.20  #  validation_split: 浮点数。Float. 保留用于验证的图像的比例（严格在0和1之间）。  
    )

    dataflow_kwargs = dict(
        target_size=self.config.params_image_size[:-1],
        batch_size=self.config.params_batch_size,
        interpolation="bilinear"
    )
    
    valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
        **datagenerator_kwargs
    )

    self.valid_generator = valid_datagenerator.flow_from_directory(
        directory=self.config.validation_data,
        shuffle=False,
        **dataflow_kwargs
    )
    
  @staticmethod
  def load_model(path:Path)->tf.keras.Model:
    return tf.keras.models.load_model(path)
  
  
  def _save_model_score(self):
    score={"loss":self.score[0],"accuracy":self.score[1]}
    save_json(path=Path("scores.json"),data=score)
  
  
  def evaluation(self):
    self.model=self.load_model(self.config.path_of_model)
    self._valid_generator()
    self.score=self.model.evaluate(self.valid_generator)
    self._save_model_score()
    

   
  
  # export MLFLOW_TRACKING_URI=https://dagshub.com/mattml/deeplearning.mlflow \
  # export MLFLOW_TRACKING_USERNAME=mattml \
  # export MLFLOW_TRACKING_PASSWORD=7a736a85c503ae97eab4d94b077898f9d9bdd61a \
   
  def log_into_mlflow(self):
    
    os.environ["MLFLOW_TRACKING_URI"]=self.config.mlflow_uri
    os.environ["MLFLOW_TRACKING_USERNAME"]=self.config.mlflow_username
    os.environ["MLFLOW_TRACKING_PASSWORD"]=self.config.mlflow_pass
    mlflow.set_registry_uri(self.config.mlflow_uri)
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    
    with mlflow.start_run():

      mlflow.log_params(self.config.all_params)
      mlflow.log_metrics(
          {"loss": self.score[0], "accuracy": self.score[1]}
      )
      
      if tracking_url_type_store != "file": 
        mlflow.tensorflow.log_model(self.model, "model", registered_model_name="VGG16Model")
      else:
        mlflow.tensorflow.log_model(self.model, "model") 





  # def log_into_mlflow(self):
  #   # mlflow.set_registry_uri(self.config.mlflow_uri) 
  #   # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme 
  #   dagshub.init(self.config.mlflow_repository, self.config.mlflow_username,mlflow=True)
   
  #   with mlflow.start_run():
  #     self._valid_generator()
  #     self.score=self.model.evaluate(self.valid_generator)
  #     mlflow.log_params(self.config.all_params)
  #     mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})
  #     mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
    
  #   mlflow.end_run()
    
    
        # Model registry does not work with file store
        # if tracking_url_type_store != "file":

        #     # Register the model
        #     # There are other ways to use the Model Registry, which depends on the use case,
        #     # please refer to the doc for more information:
        #     # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        #     mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
        # else:
        #     mlflow.keras.log_model(self.model, "model")
        
        
        
        
  #     def evaluation(self):
  #   self.model=self.load_model(self.config.path_of_model)
  #   self._valid_generator()
  #   self.score=self.model.evaluate(self.valid_generator)
    
  # def save_model_evaluation_score(self):
  #   score={"loss":self.score[0],"accuracy":self.score[1]}
  #   save_json(path=Path("scores.json"),data=score)