from dataclasses import dataclass
from pathlib import Path  
 
@dataclass(frozen=True) #表示这是不可变对象，初始化后不能重新赋值
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file:Path
    unzip_dir:Path  