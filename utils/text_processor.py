# utils/text_processor.py
class TextChunker:
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 4000, overlap: int = 500) -> list:
        """
        텍스트를 chunk_size 만큼 분할하며, 문맥 단절을 막기 위해 overlap 만큼 겹치게 자름.
        16GB VRAM 기준 7B 모델은 한 번에 4000~6000자 내외가 안전함.
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunks.append(text[start:end])
            if end == text_length:
                break
            start += chunk_size - overlap
            
        return chunks
    
    