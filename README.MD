# LLM_Qwen
# Multi-Agent Evaluation System 

## 🎯 Project Overview

단순한 프롬프팅을 넘어, 이기종 데이터(기획서 텍스트, 콘셉트 아트/영상)를 처리하기 위해 두 개의 특화된 Qwen 2.5 에이전트를 교차 활용합니다.
- **Text Agent (`Qwen2.5-7B-Instruct`)**: RAG 시스템 및 다중 에이전트(Live-Ops PD vs Package PD) 추론용. 서사적 밀도 및 캐릭터 확장성을 분석합니다.
- **Vision Agent (`Qwen2.5-VL-7B-Instruct`)**: 게임 스크린샷 및 아트 스타일을 분석하여 도메인 특화 메타 지표를 추출합니다.

## ⚙️ Architecture & Features
* **Config-Driven Architecture**: `config.yaml`을 통해 모델 메타데이터 및 추론 환경을 관리하여 유연한 확장성을 제공합니다.

* **VRAM Optimization**: 16GB VRAM 환경에서의 제약을 극복하기 위해 `bfloat16` 양자화를 적용하고, 에이전트 스위칭 간 명시적 메모리 해제(Garbage Collection 및 Cache Emptying) 유틸리티를 구현했습니다.

## Development Environment
- GPU: 5060ti - 16GB


## 🚀 Getting Started

### Prerequisites
```bash
conda create -n qwen_env python=3.10
conda activate qwen_env
pip install -r requirements.txt


📝 TODO (Future Works)
[ ] LangGraph 기반의 Multi-Agent 상태 유지(Stateful) 라우팅 로직 연동
[ ] Neo4j 연동을 통한 GraphRAG Cypher 쿼리 파이프라인 통합
