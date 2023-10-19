from dataclasses import dataclass
from pathlib import Path  
 
@dataclass(frozen=True) #表示这是不可变对象，初始化后不能重新赋值
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file:Path
    unzip_dir:Path  
    
    
@dataclass(frozen=True)
class PrepareBaseModelConfig:
  root_dir:Path
  base_model_path:Path
  updated_base_model_path:Path
  params_image_size:list
  params_learning_rate:float
  params_include_top:bool
  params_weights:str
  params_classes:int
  
@dataclass(frozen=True)
class PrepareCallbacksConfig:
  root_dir: str
  tensorboard_root_log_dir: str
  checkpoint_model_filepath: str
  
  
@dataclass(frozen=True)
class TrainingConfig:
  root_dir: Path
  trained_model_path: Path
  updated_base_model_path: Path
  training_data: Path
  validation_data: Path
  params_epochs:int
  params_batch_size:int
  params_is_augmentation:bool
  params_image_size:list
  
  
@dataclass(frozen=True)
class EvaluationConfig:
  path_of_model: Path
  validation_data: Path
  all_params: dict
  params_batch_size:int
  params_image_size:list
  mlflow_repository: str
  mlflow_username:str
  mlflow_pass:str
  mlflow_uri:str  