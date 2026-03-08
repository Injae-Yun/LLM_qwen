# main.py
from models.model_loader import ModelFactory

def main(target_agent="text_agent"):
    factory = ModelFactory("config.yaml")

    # 1. 에이전트 로드 및 테스트
    model, tokenizer = factory.load_model(target_agent)
    # (추론 로직 수행...)
    
    # 2. VRAM 확보 (16GB 환경에서 두 개를 동시에 올리기 버거울 경우)
    ModelFactory.clear_vram(model)
    


if __name__ == "__main__":
    # target_agent는 "text_agent" 또는 "vision_agent"로 설정 가능
    main(target_agent="text_agent")

