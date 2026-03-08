import json

class PromptManager:
    @staticmethod
    def get_extraction_prompt(raw_text: str, chunk_index: int, total_chunks: int, previous_state: dict = None, glossary: dict = None) -> list:
        
        state_context = ""
        if previous_state and "Entities with Timeline" in previous_state:
            state_context = "\n[Summary of Entity States up to Previous Chapter]\n"
            state_context += json.dumps(previous_state, ensure_ascii=False, indent=2)
            state_context += "\nRefer to this state. Maintain the exact 'Root ID' for existing entities, and ONLY extract new, significant state changes."

        # 1. Glossary 컨텍스트 주입 로직 추가
        glossary_context = ""
        if glossary:
            glossary_context = "\n[GLOBAL GLOSSARY - STRICTLY USE THESE TERMS]\n"
            glossary_context += json.dumps(glossary, ensure_ascii=False, indent=2)

        json_schema = {
            "Entities with Timeline": [
                {
                    "Root ID": "Unique ID (e.g., LastInnKeeper)",
                    "Name": "Entity Name (Must exact match the Glossary if present)",
                    "Core Identity": "Core identity or profession (Use Glossary term)",
                    "State Timeline": [
                        {
                            "Node ID": "Versioned Node ID (e.g., LastInnKeeper_v1)",
                            "Event ID": f"Ch_{chunk_index}_[Event_Summary]", 
                            "Concept": "Current concept or role",
                            "Visual": "Visual characteristics",
                            "Trigger Event": "The major event that caused this state change"
                        }
                    ]
                }
            ]
        }

        # 2. 규칙에 Glossary 참조 및 원어(Korean) 보존 강제 조항 추가
        system_instruction = f"""You are a Temporal Knowledge Graph Data Pipeline Agent for game IP analysis.
The user will provide text which is part {chunk_index} of {total_chunks} of the entire story.{state_context}{glossary_context}

[Extraction Rules]
1. If an entity from the previous state appears, strictly reuse their existing 'Root ID'.
2. [CRITICAL] DO NOT create new nodes for trivial actions, dialogue, mere observations, or temporary emotional changes.
3. ONLY create a new Node in the 'State Timeline' when an entity undergoes a MAJOR, PERMANENT change (e.g., significant visual/equipment change, faction change, death, awakening, or fundamental role change).
4. If an entity appears in the text but does not undergo a major change as defined in Rule 3, DO NOT add a new timeline node for them.
5. 'Event ID' MUST start exactly with 'Ch_{chunk_index}_'.
6. [GLOSSARY RESPECT] If an entity, faction, or race exists in the [GLOBAL GLOSSARY], you MUST use the exact string from the glossary for the 'Name' and 'Core Identity' values. If the glossary term is in Korean, copy the Korean text exactly. DO NOT translate or romanize it.
7. Strictly adhere to the provided JSON schema. Output ONLY valid JSON, without any markdown formatting, preambles, or explanations.

[Output JSON Schema]
{json.dumps(json_schema, ensure_ascii=False, indent=2)}"""

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Analyze the following text from chapter {chunk_index} and extract the JSON data:\n\n{raw_text}"}
        ]
        
        return messages

    @staticmethod
    def get_ner_prompt(raw_text: str) -> list:
        json_schema = {
            "Glossary": {
                "Characters": ["Exact character names from text"],
                "Factions_and_Groups": ["Guilds, organizations, families"],
                "Races_and_Classes": ["Elves, Warriors, specific job titles"],
                "Locations": ["Cities, dungeons, specific places"],
                "Key_Items_and_Concepts": ["Unique weapons, magic terms, specific artifacts"]
            }
        }

        # 1. 메타 정보 배제 규칙(Rule 3) 추가 및 중복된 Rule 2 제거
        system_instruction = f"""You are a strict Named Entity Recognition (NER) Agent for a game IP database.
Your ONLY task is to extract proper nouns and key terminology from the provided text to build a definitive glossary.

[Extraction Rules]
1. [EXACT MATCH ONLY] You MUST extract the terms EXACTLY as they appear in the source text. Do not translate them to English. If the text is in Korean, extract the Korean substrings perfectly.
2. [NO PARAPHRASING] Do not alter, summarize, or explain the terms.
3. [EXCLUDE META-INFO] DO NOT extract the names of authors, pen names, illustrators, publishers, or any real-world entities associated with the creation of the text. Only extract in-universe fictional entities.
4. [EXCLUSION] Do not extract generic words (e.g., "sword", "village", "man"). Only extract specific proper nouns or unique IP terminology.
5. Output ONLY valid JSON adhering strictly to the schema.

[Output JSON Schema]
{json.dumps(json_schema, ensure_ascii=False, indent=2)}"""

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Extract the glossary from the following text:\n\n{raw_text}"}
        ]
        
        return messages