import json

class PromptManager:
    @staticmethod
    def get_extraction_prompt(raw_text: str, chunk_index: int, total_chunks: int, previous_state: dict = None) -> list:
        
        state_context = ""
        if previous_state and "Entities with Timeline" in previous_state:
            state_context = "\n[Summary of Entity States up to Previous Chapter]\n"
            state_context += json.dumps(previous_state, indent=2)
            state_context += "\nRefer to this state. Maintain the exact 'Root ID' for existing entities, and ONLY extract new, significant state changes."

        json_schema = {
            "Entities with Timeline": [
                {
                    "Root ID": "Unique ID (e.g., LastInnKeeper)",
                    "Name": "Entity Name",
                    "Core Identity": "Core identity or profession",
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

        system_instruction = f"""You are a Temporal Knowledge Graph Data Pipeline Agent for game IP analysis.
The user will provide text which is part {chunk_index} of {total_chunks} of the entire story.{state_context}

[Extraction Rules]
1. If an entity from the previous state appears, strictly reuse their existing 'Root ID'.
2. [CRITICAL] DO NOT create new nodes for trivial actions, dialogue, mere observations, or temporary emotional changes.
3. ONLY create a new Node in the 'State Timeline' when an entity undergoes a MAJOR, PERMANENT change (e.g., significant visual/equipment change, faction change, death, awakening, or fundamental role change).
4. If an entity appears in the text but does not undergo a major change as defined in Rule 3, DO NOT add a new timeline node for them.
5. 'Event ID' MUST start exactly with 'Ch_{chunk_index}_'.
6. Strictly adhere to the provided JSON schema. Output ONLY valid JSON, without any markdown formatting, preambles, or explanations.

[Output JSON Schema]
{json.dumps(json_schema, indent=2)}"""

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Analyze the following text from chapter {chunk_index} and extract the JSON data:\n\n{raw_text}"}
        ]
        
        return messages