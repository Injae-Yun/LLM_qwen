# main.py
from models.model_loader import ModelFactory
from utils.prompt_manager import PromptManager 
from utils.text_processor import TextChunker 
import json
import torch
import process # process 모듈에서 run_text_agent와 run_vision_agent 함수를 임포트

def main(target_agent="text_agent"):
    factory = ModelFactory("config.yaml")

    # 1. 에이전트 로드
    model, processor = factory.load_model(target_agent)
    
    # 2. 추론 로직 수행 (테스트용)
    if target_agent == "text_agent":
        # 1. 테스트용 비정형 입력 데이터 (기획서 5페이지 서사 기반)
        # txt 파일 load
        file_path = "data/sample_novel.txt" # utils/data_loader.py의 load_txt 함수로 txt 파일 로드 (예시)
        #process.GlossaryManager(model, processor, file_path=file_path) # 용어집 추출 테스트
        process.run_text_agent(model, processor, file_path=file_path)

    elif target_agent == "vision_agent":
        # Qwen-VL은 qwen_vl_utils 라이브러리가 필요합니다.
        from qwen_vl_utils import process_vision_info
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
                {"type": "text", "text": "이 이미지의 아트 스타일을 분석해줘."}
            ]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=256)
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"[{target_agent} 응답]\n{response}")
    
    # 3. VRAM 확실한 확보 (명시적 참조 제거)
    del model
    del processor
    ModelFactory.clear_vram()

if __name__ == "__main__":
    # 텍스트 에이전트와 비전 에이전트를 번갈아가며 테스트
    target = "text_agent"  # text_agent 또는 vision_agent
    main(target_agent=target) # 번갈아가며 테스트