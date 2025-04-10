import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import json
import yaml
import logging
from pathlib import Path

def setup_logging(log_path: str = "logs") -> None:
    """로깅 설정 함수"""
    Path(log_path).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_path}/app.log"),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """설정 파일 로드 함수"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_json(data: Dict[str, Any], path: str) -> None:
    """JSON 파일 저장 함수"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_json(path: str) -> Dict[str, Any]:
    """JSON 파일 로드 함수"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_directory(path: str) -> None:
    """디렉토리 생성 함수"""
    Path(path).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # 테스트 코드 작성
    pass 