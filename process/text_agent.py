import json
import torch
import re
from utils.prompt_manager import PromptManager
from utils.text_processor import TextChunker
from models.model_loader import ModelFactory

def run_text_agent(model, processor, file_path="data/sample_novel.txt"):
    try:
        with open(file_path, 'r',encoding='cp949') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    #1.5 텍스트 청킹 (모델 입력 길이 제한 고려)
    chunks = TextChunker.chunk_text(content, chunk_size=5000, overlap=500)
    # 실제로는 chunk를 문맥 단위로 분할하는 것이 필요함 (예 챕터별, 섹션별 등) - 여기서는 단순 문자 수 기반 청킹으로 예시를 들었음
    print(f"총 {len(content)}자 텍스트가 {len(chunks)}개의 청크로 분할되었습니다.")

    extracted_graphs = []
    current_state_feedback = None  # 이전 청크까지의 추출 결과를 피드백으로 활용하기 위한 변수
    # 3. 청크 단위로 순차 추출
    for i, chunk in enumerate(chunks):
        print(f"\n--- [청크 {i+1}/{len(chunks)}] 처리 중 ---")
        if i> 30:  # 테스트용으로 최대 30개 청크까지만 처리
            print("테스트용으로 30개 청크까지만 처리합니다.")
            break
        messages = PromptManager.get_extraction_prompt(
            raw_text=chunk, 
            chunk_index=i, 
            total_chunks=len(chunks),
            previous_state=current_state_feedback
        )         
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = processor([text], return_tensors="pt").to(model.device)
        
        # 추론
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs, 
                max_new_tokens=1024,
                do_sample=False
            )
        
        input_length = model_inputs['input_ids'].shape[1]
        response = processor.decode(generated_ids[0][input_length:], skip_special_tokens=True)
        
        # JSON 파싱
        try:
            clean_json_str = response.replace("```json", "").replace("```", "").strip()
            match = re.search(r'\{.*\}', clean_json_str, re.DOTALL)
            parsed_data = json.loads(match.group())
            extracted_graphs.append(parsed_data)
            current_state_feedback = parsed_data
            print(f"청크 {i+1} 파싱 성공.")
        except json.JSONDecodeError:
            print(f"청크 {i+1} 파싱 실패 (Raw Text 참조).")
        
        # 4. 필수: 루프마다 VRAM 강제 초기화 (OOM 방지)
        del model_inputs
        del generated_ids
        torch.cuda.empty_cache()

    # 5. 최종 지식 그래프 병합 (Entity Consolidation)
    merged_knowledge_graph = {"Entities with Timeline": {}}

    for graph_chunk in extracted_graphs:
        if not graph_chunk or "Entities with Timeline" not in graph_chunk:
            continue
            
        for entity in graph_chunk["Entities with Timeline"]:
            root_id = entity.get("Root ID")
            if not root_id:
                continue
            
            # 처음 등장하는 엔티티인 경우 초기화
            if root_id not in merged_knowledge_graph["Entities with Timeline"]:
                merged_knowledge_graph["Entities with Timeline"][root_id] = {
                    "Root ID": root_id,
                    "Name": entity.get("Name", ""),
                    "Core Identity": entity.get("Core Identity", ""),
                    "State Timeline": []
                }
            
            # 이미 존재하는 엔티티라면 타임라인(Event)만 연장(Append)
            new_events = entity.get("State Timeline", [])
            merged_knowledge_graph["Entities with Timeline"][root_id]["State Timeline"].extend(new_events)

    # 딕셔너리 구조를 다시 리스트 스키마로 변환
    final_output = {
        "Entities with Timeline": list(merged_knowledge_graph["Entities with Timeline"].values())
    }

    # 6. 최종 추출 결과 저장
    with open("extracted_knowledge_graph.json", "w", encoding="utf-8") as out_f:
        json.dump(final_output, out_f, indent=2, ensure_ascii=False)
        
    print("\n최종 지식 그래프 병합 및 저장이 완료되었습니다. (extracted_knowledge_graph.json)")
    del model
    del processor
    ModelFactory.clear_vram()