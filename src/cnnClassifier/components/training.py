import os
import zipfile
import tensorflow as tf
import urllib.request as request
import time
from cnnClassifier.config.configuration import TrainingConfig
from pathlib import Path


class Training:
  def __init__(self,config:TrainingConfig):
    self.config = config
    
  def get_base_model(self):
    self.model=tf.keras.models.load_model(
      self.config.updated_base_model_path
    )
       
  def train_valid_generator(self):   
# rescale: 重缩放因子。默认为 None。如果是 None 或 0，不进行缩放，否则将数据乘以所提供的值（在应用任何其他转换之前）。
# rescale的作用是对图片的每个像素值均乘上这个放缩因子，这个操作在所有其它变换操作之前执行，在一些模型当中，
# 直接输入原图的像素值可能会落入激活函数的“死亡区”，因此设置放缩因子为1/255，把像素值放缩到0和1之间有利于模型的收敛，避免神经元“死亡”。 

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
    
    train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=40,
        horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        **datagenerator_kwargs
    )   

    self.train_generator = train_datagenerator.flow_from_directory(
        directory=self.config.training_data,
        shuffle=True,
        **dataflow_kwargs
    )  
    
  @staticmethod
  def save_model(path:Path, model:tf.keras.Model):
    model.save(path)
    
  def train(self,callback_list:list):
    self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
    self.validition_steps = self.valid_generator.samples // self.valid_generator.batch_size
    
    self.model.fit(
        self.train_generator,
        epochs = self.config.params_epochs,
        steps_per_epoch=self.steps_per_epoch,
        validation_data=self.valid_generator,
        validation_steps=self.validition_steps,
        callbacks=callback_list
        )
    
    self.save_model(self.config.trained_model_path,self.model)