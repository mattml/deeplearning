import os
import zipfile
import tensorflow as tf
from pathlib import Path
import time
from cnnClassifier.entity.config_entity import PrepareCallbacksConfig

class PrepareCallbacks:
  def __init__(self,config:PrepareCallbacksConfig):
    self.config = config
    
  @property
  def _create_tb_callbacks(self):
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    tb_running_log_dir =os.path.join(
      self.config.tensorboard_root_log_dir,
      f"tb_logs_at_{timestamp}"      
    )
    return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)
  
  @property
  def _create_checkpoint_callbacks(self):
    checkpoint_callbacks=tf.keras.callbacks.ModelCheckpoint(
      filepath=self.config.checkpoint_model_filepath,
      save_best_only=True)
    return checkpoint_callbacks
    
  def get_tb_ckpt_callbacks(self):
    return [
      self._create_tb_callbacks, 
      self._create_checkpoint_callbacks
      ]
                                    