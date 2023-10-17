import os
import urllib.request as request
import zipfile
import tensorflow as tf
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path

class PrepareBasemModel:
  def __init__(self,config:PrepareBaseModelConfig):
    self.config = config
    
  def get_base_model(self): # 用配置文件里的参数建立基础模型
    #输入尺寸224,224,3 在imagenet上预训练过，不要最后的全连接层
    self.model=tf.keras.applications.vgg16.VGG16(
      include_top=self.config.params_include_top,
      weights=self.config.params_weights,
      input_shape=self.config.params_image_size
    )
    
    self.save_model(self.config.base_model_path,model=self.model)
    
  @staticmethod
  def save_model(path: Path, model: tf.keras.Model):
    model.save(path)
    
  @staticmethod  #模型微调，最后一层拉平，加入2个输出的全连接层。
  def _prepare_full_model(model: tf.keras.Model,num_classes:int,
                          freeze_all:bool,freeze_till:int,learning_rate:float):
    if freeze_all:
      for layer in model.layers:
        layer.trainable = False
    elif(freeze_till is not None) and (freeze_till>0):
      for layer in model.layers[:-freeze_till]:
        layer.trainable = False
        
    flatten_in=tf.keras.layers.Flatten()(model.output)
    
    predication=tf.keras.layers.Dense(
      units=num_classes,
      activation='softmax'
      )(flatten_in)
    
    full_model=tf.keras.models.Model(
      inputs=model.input,
      outputs=predication
      )
    
    full_model.compile(
      optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
      loss=tf.keras.losses.CategoricalCrossentropy(),
      metrics=['accuracy']
    )
    
    full_model.summary()
    return full_model
  
  def update_base_model(self):
    self.full_model=self._prepare_full_model(
      model=self.model,
      num_classes=self.config.params_classes,
      freeze_all=True,
      freeze_till=None,
      learning_rate=self.config.params_learning_rate
    )
    
    self.save_model(self.config.updated_base_model_path,model=self.full_model)
    
  
    