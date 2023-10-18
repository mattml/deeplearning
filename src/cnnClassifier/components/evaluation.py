import os
import zipfile
import tensorflow as tf
import urllib.request as request
import time
from cnnClassifier.config.configuration import EvaluationConfig
from pathlib import Path
from cnnClassifier.utils.common import save_json


class Evaluation:
  def __init__(self,config=EvaluationConfig):
    self.config = config
    
  def _valid_generator(self):
    
    datagenerator_kwargs = dict(
        rescale = 1./255,
        # validation_split=0.20  #  validation_split: 浮点数。Float. 保留用于验证的图像的比例（严格在0和1之间）。  
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
  
  def evaluation(self):
    self.model=self.load_model(self.config.path_of_model)
    self._valid_generator()
    self.score=self.model.evaluate(self.valid_generator)
    
  def save_model_evaluation_score(self):
    score={"loss":self.score[0],"accuracy":self.score[1]}
    save_json(path=Path("scores.json"),data=score)