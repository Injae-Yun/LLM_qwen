# model_loader.py
import yaml
import torch
import gc
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoProcessor, 
    Qwen2_5_VLForConditionalGeneration
)

class ModelFactory:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    def load_model(self, agent_name):
        """
        config.yaml에 정의된 agent_name을 기반으로 모델과 토크너/프로세서를 반환
        """
        cfg = self.config['models'].get(agent_name)
        if not cfg:
            raise ValueError(f"Config에 '{agent_name}'가 정의되어 있지 않습니다.")

        # dtype 매핑
        dtype = torch.bfloat16 if cfg['torch_dtype'] == "bfloat16" else torch.float16

        print(f"[{agent_name}] {cfg['model_id']} 로딩을 시작합니다...")

        if cfg['model_type'] == 'text':
            model = AutoModelForCausalLM.from_pretrained(
                cfg['model_id'],
                torch_dtype=dtype,
                device_map=cfg['device_map']
            )
            tokenizer = AutoTokenizer.from_pretrained(cfg['model_id'])
            return model, tokenizer

        elif cfg['model_type'] == 'vision':
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                cfg['model_id'],
                torch_dtype=dtype,
                device_map=cfg['device_map']
            )
            processor = AutoProcessor.from_pretrained(cfg['model_id'])
            return model, processor

        else:
            raise ValueError(f"지원하지 않는 모델 타입입니다: {cfg['model_type']}")

    @staticmethod
    def clear_vram(model_obj):
        """
        16GB VRAM 환경에서 두 모델을 스위칭할 때 메모리 누수를 방지하는 유틸리티
        """
        del model_obj
        gc.collect()
        torch.cuda.empty_cache()
        print("VRAM 해제 완료.")