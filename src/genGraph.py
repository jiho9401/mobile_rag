import argparse
import json
import os
import time
import requests
from datetime import datetime
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import random
from tqdm.auto import tqdm

class GraphGenerator:
    """LLM을 활용한 지식 그래프 생성기"""
    
    def __init__(self, server_url, llm_model="gemma3:27b", embedding_model=None):
        """초기화: LLM 서버 설정 및 임베딩 모델 로드"""
        self.server_url = server_url
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        
        print(f"📊 그래프 생성기 초기화...")
        print(f"   LLM 서버: {server_url}")
        print(f"   LLM 모델: {llm_model}")
        
        # 임베딩 모델 로드 (있는 경우)
        self.ort_session = None
        self.tokenizer = None
        
        if embedding_model:
            try:
                print(f"🧠 ONNX 임베딩 모델 로드 중: {embedding_model}")
                self.ort_session = ort.InferenceSession(embedding_model)
                self.tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
                print("   ONNX 모델 로드 완료")
            except Exception as e:
                print(f"⚠️ ONNX 모델 로드 실패: {e}")
                print("   임베딩 없이 진행합니다.")
    
    def compute_embedding(self, text):
        """텍스트 임베딩 계산 (E5 모델 형식 지원)"""
        if self.ort_session is None or self.tokenizer is None:
            return None
        
        # 빈 텍스트 체크
        if not text or text.strip() == "":
            return None
        
        try:
            # E5 모델 형식으로 텍스트 포맷팅
            formatted_text = f"passage: {text}"
            
            # 토큰화
            inputs = self.tokenizer(
                formatted_text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np"
            )
            
            # 모델 입력 준비
            model_inputs = {
                "input_ids": inputs["input_ids"]
            }
            
            # 어텐션 마스크가 필요하면 추가
            if "attention_mask" in [inp.name for inp in self.ort_session.get_inputs()]:
                model_inputs["attention_mask"] = inputs["attention_mask"]
            
            # 토큰 타입 ID가 필요하면 추가
            if "token_type_ids" in [inp.name for inp in self.ort_session.get_inputs()]:
                if "token_type_ids" in inputs:
                    model_inputs["token_type_ids"] = inputs["token_type_ids"]
                else:
                    model_inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])
            
            # ONNX 모델로 임베딩 계산
            outputs = self.ort_session.run(None, model_inputs)
            
            # 출력 형태에 따라 임베딩 추출
            if len(outputs[0].shape) == 3:  # [batch, seq_len, hidden]
                embedding = outputs[0][:, 0, :]  # CLS 토큰 임베딩
            else:  # [batch, hidden]
                embedding = outputs[0]  # 이미 풀링된 임베딩
            
            # L2 정규화
            norm = np.linalg.norm(embedding, axis=1, keepdims=True)
            normalized_embedding = embedding / norm
            
            return normalized_embedding[0].tolist()  # 배치 차원 제거 및 리스트 변환
        except Exception as e:
            print(f"⚠️ 임베딩 계산 오류: {e}")
            return None
    
    def call_llm(self, prompt):
        """LLM 호출하여 응답 생성"""
        url = f"{self.server_url}/api/generate"
        
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.7,  # 창의성을 위해 온도 높임
            "max_tokens": 2048
        }
        
        try:
            # 타임아웃 설정 추가 (180초)
            response = requests.post(url, json=payload, timeout=180)
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "API 응답 오류: 응답 필드를 찾을 수 없습니다.")
            else:
                return f"API 호출 오류: 상태 코드 {response.status_code}"
        except requests.exceptions.Timeout:
            return "API 호출 타임아웃: 서버 응답이 너무 오래 걸립니다."
        except Exception as e:
            return f"API 호출 중 예외 발생: {str(e)}"
    
    def generate_domain_knowledge(self, domain, num_nodes=500):
        """특정 도메인에 대한 지식 엔티티 생성"""
        prompt = f"""다음 도메인에 대한 핵심 개념, 용어, 인물, 기관 등을 JSON 형식으로 생성해주세요: {domain}

각 엔티티는 다음 구조를 가져야 합니다:

{{
"id": "고유 식별자",
"name": "엔티티 이름",
"type": "개념/인물/기관/기술 등 타입",
"description": "상세 설명",
"properties": {{
"속성1": "값1",
"속성2": "값2"
}}
}}


가능한 많은 다양한 엔티티를 생성해주세요. 엔티티들은 해당 도메인의 핵심적인 요소들을 포함해야 합니다.
결과는 JSON 배열 형식으로 반환해주세요.
"""
        
        print(f"🧠 '{domain}' 도메인에 대한 지식 엔티티 생성 중...")
        
        # 진행 상황 표시
        with tqdm(total=1, desc="LLM 엔티티 생성", unit="요청") as pbar:
            response = self.call_llm(prompt)
            pbar.update(1)
        
        try:
            # JSON 부분만 추출 (마크다운 코드 블록에서)
            json_text = response
            if "```json" in response:
                json_text = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_text = response.split("```")[1].split("```")[0].strip()
            
            # JSON 파싱
            entities = json.loads(json_text)
            print(f"✅ {len(entities)}개 엔티티 생성 완료")
            return entities
        except Exception as e:
            print(f"⚠️ JSON 파싱 오류: {e}")
            print("원본 응답:")
            print(response)
            return []
    
    def generate_relationships(self, entities):
        """엔티티 간 관계 생성"""
        # 엔티티 목록 준비
        entity_list = []
        for entity in entities:
            entity_info = f"ID: {entity['id']}, 이름: {entity['name']}, 유형: {entity['type']}"
            entity_list.append(entity_info)
        
        entity_text = "\n".join(entity_list)
        
        prompt = f"""다음은 지식 그래프에 포함된 엔티티 목록입니다:

{entity_text}

위 엔티티들 간의 의미 있는 관계를 JSON 형식으로 생성해주세요. 관계는 다음 구조를 가져야 합니다:

{{
"source": "시작 엔티티 ID",
"target": "대상 엔티티 ID",
"relation": "관계 유형 (예: '포함한다', '영향을 미친다', '개발했다' 등)",
"weight": 0.1과 1.0 사이의 관계 강도,
"properties": {{
"속성1": "값1",
"속성2": "값2"
}}
}}


최소 {len(entities) * 2}개의 다양한 관계를 생성해주세요. 엔티티들의 특성을 고려하여 의미 있는 관계를 만들어주세요.
결과는 JSON 배열 형식으로 반환해주세요.
"""
        
        print(f"🔄 엔티티 간 관계 생성 중...")
        
        # 진행 상황 표시
        with tqdm(total=1, desc="LLM 관계 생성", unit="요청") as pbar:
            response = self.call_llm(prompt)
            pbar.update(1)
        
        try:
            # JSON 부분만 추출 (마크다운 코드 블록에서)
            json_text = response
            if "```json" in response:
                json_text = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_text = response.split("```")[1].split("```")[0].strip()
            
            # JSON 파싱
            relationships = json.loads(json_text)
            print(f"✅ {len(relationships)}개 관계 생성 완료")
            return relationships
        except Exception as e:
            print(f"⚠️ JSON 파싱 오류: {e}")
            print("원본 응답:")
            print(response)
            return []
    
    def validate_knowledge_graph(self, nodes, edges):
        """지식 그래프의 유효성 검증"""
        print("🔍 지식 그래프 유효성 검증 중...")
        
        # 노드 ID 목록 생성
        node_ids = set(node["id"] for node in nodes)
        
        # 엣지 검증
        valid_edges = []
        invalid_edges = []
        
        for edge in tqdm(edges, desc="관계 검증", unit="관계"):
            source = edge.get("source")
            target = edge.get("target")
            
            # 소스와 타겟이 노드 목록에 있는지 확인
            if source in node_ids and target in node_ids:
                valid_edges.append(edge)
            else:
                invalid_edges.append(edge)
        
        if invalid_edges:
            print(f"⚠️ {len(invalid_edges)}개의 유효하지 않은 관계가 제거되었습니다.")
        
        print(f"✅ 검증 완료: {len(valid_edges)}개의 유효한 관계")
        return valid_edges
    
    def add_embeddings_to_nodes(self, nodes):
        """노드에 임베딩 추가"""
        if self.ort_session is None:
            print("⚠️ 임베딩 모델이 로드되지 않아 임베딩을 추가하지 않습니다.")
            return nodes
        
        print("🧮 노드에 임베딩 추가 중...")
        
        for node in tqdm(nodes, desc="임베딩 추가", unit="노드"):
            # 임베딩 계산을 위한 텍스트 구성
            text = f"{node['name']}. {node['description']}"
            
            # 임베딩 계산
            embedding = self.compute_embedding(text)
            
            # 노드에 임베딩 추가
            if embedding is not None:
                node["embedding"] = embedding
        
        # 임베딩이 추가된 노드 수 계산
        nodes_with_embedding = sum(1 for node in nodes if "embedding" in node)
        print(f"✅ {nodes_with_embedding}/{len(nodes)} 노드에 임베딩 추가 완료")
        
        return nodes
    
    def split_text_into_chunks(self, text, max_chunk_size=1500, overlap=100):
        """텍스트를 여러 청크로 분할"""
        chunks = []
        
        # 1. 먼저 다양한 구분선으로 분할 시도
        separators = [
            "*" * 80,  # 80개의 별표
            "=" * 80,  # 80개의 등호
            "-" * 80,  # 80개의 하이픈
        ]
        
        # 구분선으로 초기 분할
        initial_chunks = [text]
        for separator in separators:
            if separator in text:
                print(f"🔍 '{separator[:10]}...' 구분선 발견")
                new_chunks = []
                for chunk in initial_chunks:
                    if separator in chunk:
                        parts = chunk.split(separator)
                        for i, part in enumerate(parts):
                            if i > 0:  # 첫 부분이 아니면 구분선을 포함시켜 맥락 유지
                                part = separator + part
                            if part.strip():
                                new_chunks.append(part.strip())
                    else:
                        if chunk.strip():
                            new_chunks.append(chunk)
                initial_chunks = new_chunks
        
        # 초기 구분선 분할 결과가 있으면 사용
        if len(initial_chunks) > 1:
            print(f"✅ 구분선 기준 {len(initial_chunks)}개 청크로 분할됨")
            chunks = initial_chunks
        else:
            # 구분선이 없거나 분할 결과가 하나뿐이면 텍스트 자체를 사용
            chunks = [text]
        
        # 2. 크기가 너무 큰 청크는 추가 분할
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_chunk_size:
                final_chunks.append(chunk)
            else:
                # 크기 기반 추가 분할
                sub_chunks = self._size_based_split(chunk, max_chunk_size, overlap)
                final_chunks.extend(sub_chunks)
        
        print(f"✅ 최종적으로 {len(final_chunks)}개 청크로 분할됨")
        return final_chunks
    
    def _size_based_split(self, text, max_chunk_size=1500, overlap=100):
        """텍스트를 크기 기반으로 분할하는 내부 함수"""
        chunks = []
        
        # 먼저 단락으로 분할 시도
        paragraphs = text.split("\n\n")
        if len(paragraphs) > 1:
            current_chunk = ""
            for para in paragraphs:
                para = para.strip()
                if not para:  # 빈 단락 건너뛰기
                    continue
                
                # 현재 청크가 비어있거나, 단락을 추가해도 최대 크기를 초과하지 않는 경우
                if not current_chunk or len(current_chunk) + len(para) + 2 <= max_chunk_size:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
                else:
                    # 현재 청크를 저장하고 새 청크 시작
                    chunks.append(current_chunk)
                    current_chunk = para
            
            # 마지막 청크 추가
            if current_chunk:
                chunks.append(current_chunk)
            
            # 청크가 잘 분할되었으면 반환
            if len(chunks) > 1:
                return chunks
        
        # 단락 분할이 효과적이지 않으면 문장 단위로 분할
        start = 0
        while start < len(text):
            end = min(start + max_chunk_size, len(text))
            
            # 다음 문장 경계까지 확장 (가능한 경우)
            if end < len(text):
                # 다양한 문장 끝 기호 찾기
                sentence_ends = []
                for punct in ['. ', '? ', '! ', '.\n', '?\n', '!\n']:
                    pos = text.rfind(punct, start, min(end + 200, len(text)))
                    if pos != -1:
                        sentence_ends.append(pos + len(punct) - 1)
                
                if sentence_ends:
                    end = max(sentence_ends) + 1
            
            # 청크 추가
            chunk_text = text[start:end].strip()
            if chunk_text:  # 빈 청크는 추가하지 않음
                chunks.append(chunk_text)
            
            # 무한 루프 방지
            if end <= start:
                start = start + max_chunk_size // 2  # 강제로 이동
            else:
                start = end - overlap  # 오버랩 적용
        
        return chunks
    
    def extract_entities_from_chunk(self, text_chunk, max_entities=100, existing_names=None):
        """텍스트 청크에서 엔티티 추출"""
        if existing_names is None:
            existing_names = set()
        
        # 너무 짧은 청크는 건너뛰기
        if len(text_chunk.strip()) < 50:
            print("⚠️ 청크가 너무 짧아 건너뜁니다.")
            return []
        
        existing_info = ""
        if existing_names:
            # 너무 많은 이름을 프롬프트에 포함하지 않도록 제한
            sample_names = list(existing_names)[:20]
            existing_info = f"\n\n주의: 다음 이름과 중복되지 않는 새로운 엔티티를 생성해주세요: {', '.join(sample_names)}"
        
        prompt = f"""다음 텍스트 청크를 분석하여 핵심 개념, 용어, 인물, 기관 등을 JSON 형식으로 생성해주세요:

{text_chunk}

각 엔티티는 다음 구조를 가져야 합니다:

{{
"id": "임시ID",
"name": "엔티티 이름",
"type": "개념/인물/기관/기술 등 타입",
"description": "상세 설명",
"properties": {{
"속성1": "값1",
"속성2": "값2"
}}
}}

가능한 많은(최소 {max_entities}개) 다양한 엔티티를 생성해주세요. 텍스트에서 명확하게 언급된 핵심적인 요소들을 포함해야 합니다.{existing_info}
결과는 JSON 배열 형식으로 반환해주세요.
"""
        
        with tqdm(total=1, desc="LLM 청크 처리", unit="요청") as pbar:
            response = self.call_llm(prompt)
            pbar.update(1)
        
        try:
            # JSON 부분만 추출 (마크다운 코드 블록에서)
            json_text = response
            if "```json" in response:
                json_text = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                parts = response.split("```")
                if len(parts) >= 3:  # 최소한 앞, 코드, 뒤 세 부분이 있어야 함
                    json_text = parts[1].strip()
                    # 'json' 태그가 있을 경우 제거
                    if json_text.startswith("json"):
                        json_text = json_text[4:].strip() 
            
            # 특수 문자나 공백 정리
            json_text = json_text.strip()
            
            # JSON 파싱 시도
            try:
                entities = json.loads(json_text)
                if not isinstance(entities, list):
                    print("⚠️ JSON이 배열 형식이 아닙니다.")
                    entities = []
                return entities
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON 파싱 오류: {e}")
                print("JSON 텍스트 처음 200자:")
                print(json_text[:200] + "..." if len(json_text) > 200 else json_text)
                # 빈 배열 반환하여 프로세스 계속 진행
                return []
        except Exception as e:
            print(f"⚠️ 청크 처리 중 예외 발생: {e}")
            return []
    
    def generate_domain_knowledge_from_text(self, text_file, num_nodes=500):
        """텍스트 파일을 기반으로 지식 엔티티 생성"""
        try:
            # 텍스트 파일 읽기
            with open(text_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"📄 텍스트 파일 '{text_file}' 로드 완료 ({len(content)} 자)")
        except Exception as e:
            print(f"❌ 텍스트 파일 읽기 실패: {e}")
            return []
        
        # 텍스트 청크로 분할
        chunks = self.split_text_into_chunks(content, max_chunk_size=2000)
        print(f"📑 텍스트를 {len(chunks)}개 청크로 분할")
        
        all_entities = []
        existing_entity_names = set()  # 중복 방지를 위한 이름 추적
        
        # 각 청크별로 엔티티 생성
        for i, chunk in enumerate(chunks):
            print(f"🔍 청크 {i+1}/{len(chunks)} 처리 중...")
            
            # 각 청크 처리에 예외 핸들링 추가
            try:
                chunk_entities = self.extract_entities_from_chunk(
                    chunk, 
                    100,  # 각 청크에서 추출할 최대 엔티티 수를 100으로 증가
                    existing_entity_names
                )
                
                # 중복 방지하면서 엔티티 추가
                for entity in chunk_entities:
                    if "name" in entity and entity["name"].lower() not in existing_entity_names:
                        all_entities.append(entity)
                        existing_entity_names.add(entity["name"].lower())
                
                print(f"✅ 현재까지 {len(all_entities)} 엔티티 생성")
                
                # 목표 노드 수 달성 체크 제거 - 모든 청크를 처리하도록 함
            except Exception as e:
                print(f"⚠️ 청크 {i+1} 처리 중 오류 발생: {e}")
                continue  # 다음 청크로 진행
        
        # 아직 목표 노드 수에 도달하지 못했고 일부 엔티티라도 생성되었으면 추가 생성 시도
        if len(all_entities) < num_nodes and len(all_entities) > 0:
            try:
                print(f"🔄 추가 엔티티 생성 중... ({len(all_entities)}/{num_nodes})")
                additional_entities = self.generate_additional_entities(
                    content, all_entities, num_nodes - len(all_entities)
                )
                
                # 중복 방지하면서 추가 엔티티 병합
                for entity in additional_entities:
                    if "name" in entity and entity["name"].lower() not in existing_entity_names:
                        all_entities.append(entity)
                        existing_entity_names.add(entity["name"].lower())
            except Exception as e:
                print(f"⚠️ 추가 엔티티 생성 중 오류 발생: {e}")
        
        if not all_entities:
            print("⚠️ 엔티티가 생성되지 않았습니다. 기본 엔티티를 생성합니다.")
            # 기본 엔티티 생성
            domain = os.path.basename(text_file)
            all_entities = self.generate_domain_knowledge(domain, min(20, num_nodes))
        
        # ID 재할당
        for i, entity in enumerate(all_entities):
            entity["id"] = f"E{i+1:03d}"
        
        print(f"✅ 총 {len(all_entities)}개 엔티티 생성 완료")
        return all_entities
    
    def generate_additional_entities(self, full_text, existing_entities, count=20):
        """기존 엔티티를 고려하여 추가 엔티티 생성"""
        # 기존 엔티티 정보 준비
        entity_names = [entity["name"] for entity in existing_entities[:30]]  # 처음 30개만
        entity_info = ", ".join(entity_names)
        
        prompt = f"""다음은 텍스트에서 이미 추출한 주요 엔티티들입니다:
{entity_info}

이제 동일한 텍스트에서 위에 나열되지 않은 새로운 엔티티를 정확히 {count}개 생성해주세요.
텍스트 맥락에 맞는 관련성 있는 개념, 용어, 인물, 기관 등을 식별해야 합니다.

각 엔티티는 다음 구조를 가져야 합니다:

{{
"id": "임시ID",
"name": "엔티티 이름 (위 목록에 없는 새로운 이름)",
"type": "개념/인물/기관/기술 등 타입",
"description": "상세 설명",
"properties": {{
"속성1": "값1",
"속성2": "값2"
}}
}}

중요: 정확히 {count}개의 새로운 엔티티를 JSON 배열 형식으로 반환해주세요.
"""
        
        with tqdm(total=1, desc="LLM 추가 엔티티 생성", unit="요청") as pbar:
            response = self.call_llm(prompt)
            pbar.update(1)
        
        try:
            # JSON 부분만 추출 (마크다운 코드 블록에서)
            json_text = response
            if "```json" in response:
                json_text = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_text = response.split("```")[1].split("```")[0].strip()
            
            # JSON 파싱
            entities = json.loads(json_text)
            return entities
        except Exception as e:
            print(f"⚠️ 추가 엔티티 생성 오류: {e}")
            return []
    
    def create_knowledge_graph(self, domain=None, text_file=None, num_nodes=500, output_file=None):
        """지식 그래프 생성 및 저장 (도메인명 또는 텍스트 파일 기반)"""
        if text_file:
            print(f"\n🌐 '{text_file}' 텍스트 파일 기반 지식 그래프 생성 시작")
            # 텍스트 파일 기반 엔티티 생성
            entities = self.generate_domain_knowledge_from_text(text_file, num_nodes)
            if domain is None:
                domain = os.path.basename(text_file)  # 파일명을 도메인명으로 사용
        else:
            print(f"\n🌐 '{domain}' 도메인의 지식 그래프 생성 시작")
            # 도메인명 기반 엔티티 생성
            entities = self.generate_domain_knowledge(domain, num_nodes)
        
        if not entities:
            print("❌ 엔티티 생성 실패")
            return None
        
        # 관계 생성
        relationships = self.generate_relationships(entities)
        if not relationships:
            print("❌ 관계 생성 실패")
            return None
        
        # 관계 유효성 검증
        valid_relationships = self.validate_knowledge_graph(entities, relationships)
        
        # 임베딩 추가 (있는 경우)
        entities_with_embeddings = self.add_embeddings_to_nodes(entities)
        
        # 지식 그래프 구성
        knowledge_graph = {
            "metadata": {
                "domain": domain,
                "created_at": datetime.now().isoformat(),
                "node_count": len(entities_with_embeddings),
                "edge_count": len(valid_relationships)
            },
            "nodes": entities_with_embeddings,
            "edges": valid_relationships
        }
        
        # 지식 그래프 저장
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(knowledge_graph, f, ensure_ascii=False, indent=2)
            print(f"✅ 지식 그래프가 {output_file}에 저장되었습니다.")
        
        print(f"✅ 지식 그래프 생성 완료: 노드 {len(entities_with_embeddings)}개, 엣지 {len(valid_relationships)}개")
        return knowledge_graph
    
    def enrich_knowledge_graph(self, graph_file, output_file=None):
        """기존 지식 그래프를 확장 및 보강"""
        print(f"\n🔄 기존 지식 그래프 보강 시작: {graph_file}")
        
        # 기존 그래프 로드
        try:
            with open(graph_file, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            nodes = graph_data.get("nodes", [])
            edges = graph_data.get("edges", [])
            metadata = graph_data.get("metadata", {})
            
            print(f"   로드된 그래프: 노드 {len(nodes)}개, 엣지 {len(edges)}개")
        except Exception as e:
            print(f"❌ 그래프 로드 실패: {e}")
            return None
        
        # 노드 목록 텍스트 구성
        node_list = []
        for node in nodes:
            node_info = f"ID: {node['id']}, 이름: {node['name']}, 유형: {node['type']}"
            node_list.append(node_info)
        
        node_text = "\n".join(node_list)
        
        # 새 노드 생성 프롬프트
        new_nodes_prompt = f"""다음은 기존 지식 그래프의 노드 목록입니다:

{node_text}

위 지식 그래프를 보강하기 위한 새로운 노드 5-10개를 JSON 형식으로 생성해주세요. 기존 노드와 관련되지만 누락된 중요한 개념이나 엔티티를 추가해주세요.
각 노드는 다음 구조를 가져야 합니다:

{{
"id": "고유 식별자 (기존 ID와 중복되지 않게)",
"name": "엔티티 이름",
"type": "개념/인물/기관/기술 등 타입",
"description": "상세 설명",
"properties": {{
"속성1": "값1",
"속성2": "값2"
}}
}}


결과는 JSON 배열 형식으로 반환해주세요.
"""
        
        # 새 노드 생성
        print("🧠 새 노드 생성 중...")
        response = self.call_llm(new_nodes_prompt)
        
        try:
            # JSON 부분만 추출
            json_text = response
            if "```json" in response:
                json_text = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_text = response.split("```")[1].split("```")[0].strip()
            
            # JSON 파싱
            new_nodes = json.loads(json_text)
            print(f"✅ {len(new_nodes)}개 새 노드 생성 완료")
            
            # 새 노드에 임베딩 추가
            new_nodes_with_embeddings = self.add_embeddings_to_nodes(new_nodes)
            
            # 기존 ID와 중복 방지
            existing_ids = set(node["id"] for node in nodes)
            valid_new_nodes = []
            
            for node in new_nodes_with_embeddings:
                if node["id"] in existing_ids:
                    # ID 충돌 시 새 ID 생성
                    node["id"] = f"n{len(existing_ids) + len(valid_new_nodes) + 1}"
                
                valid_new_nodes.append(node)
                existing_ids.add(node["id"])
            
            # 모든 노드 목록 업데이트
            all_nodes = nodes + valid_new_nodes
            
            # 새 관계 생성 준비
            all_node_list = []
            for node in all_nodes:
                node_info = f"ID: {node['id']}, 이름: {node['name']}, 유형: {node['type']}"
                all_node_list.append(node_info)
            
            all_node_text = "\n".join(all_node_list)
            
            # 새 관계 생성 프롬프트
            new_edges_prompt = f"""다음은 업데이트된 지식 그래프의 노드 목록입니다:

{all_node_text}

기존 노드와 새 노드 간의 새로운 관계 10-15개를 JSON 형식으로 생성해주세요. 특히 새로 추가된 노드가 기존 노드와 어떻게 연결되는지 표현해주세요.
각 관계는 다음 구조를 가져야 합니다:

{{
"source": "시작 엔티티 ID",
"target": "대상 엔티티 ID",
"relation": "관계 유형",
"weight": 0.1과 1.0 사이의 관계 강도,
"properties": {{
"속성1": "값1",
"속성2": "값2"
}}
}}


결과는 JSON 배열 형식으로 반환해주세요.
"""
            
            # 새 관계 생성
            print("🔄 새 관계 생성 중...")
            response = self.call_llm(new_edges_prompt)
            
            try:
                # JSON 부분만 추출
                json_text = response
                if "```json" in response:
                    json_text = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    json_text = response.split("```")[1].split("```")[0].strip()
                
                # JSON 파싱
                new_edges = json.loads(json_text)
                print(f"✅ {len(new_edges)}개 새 관계 생성 완료")
                
                # 관계 유효성 검증
                valid_new_edges = self.validate_knowledge_graph(all_nodes, new_edges)
                
                # 모든 관계 목록 업데이트
                all_edges = edges + valid_new_edges
                
                # 메타데이터 업데이트
                updated_metadata = metadata.copy()
                updated_metadata["updated_at"] = datetime.now().isoformat()
                updated_metadata["node_count"] = len(all_nodes)
                updated_metadata["edge_count"] = len(all_edges)
                updated_metadata["enrichment_count"] = updated_metadata.get("enrichment_count", 0) + 1
                
                # 업데이트된 그래프 구성
                updated_graph = {
                    "metadata": updated_metadata,
                    "nodes": all_nodes,
                    "edges": all_edges
                }
                
                # 업데이트된 그래프 저장
                if not output_file:
                    output_file = graph_file
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(updated_graph, f, ensure_ascii=False, indent=2)
                
                print(f"✅ 보강된 지식 그래프가 {output_file}에 저장되었습니다.")
                print(f"   총 노드: {len(all_nodes)}개 (새 노드: {len(valid_new_nodes)}개)")
                print(f"   총 관계: {len(all_edges)}개 (새 관계: {len(valid_new_edges)}개)")
                
                return updated_graph
            except Exception as e:
                print(f"❌ 새 관계 생성 실패: {e}")
                return None
        except Exception as e:
            print(f"❌ 새 노드 생성 실패: {e}")
            return None

def main():
    """메인 함수"""
    # 명령줄 인터페이스 설정
    parser = argparse.ArgumentParser(description="LLM 기반 지식 그래프 생성기")
    parser.add_argument("--server", "-s", help="LLM 서버 URL")
    parser.add_argument("--llm", "-l", help="LLM 모델 이름", default="gemma3:27b")
    parser.add_argument("--domain", "-d", help="지식 그래프 도메인", default="대학교 안내")
    parser.add_argument("--nodes", "-n", type=int, help="생성할 노드 수", default=1000)
    parser.add_argument("--output", "-o", help="출력 JSON 파일 경로", default="knowledge_graph.json")
    parser.add_argument("--embedding", "-e", help="ONNX 임베딩 모델 파일 경로")
    parser.add_argument("--enrich", "-r", help="보강할 기존 그래프 파일 경로")
    parser.add_argument("--text", "-t", help="지식 그래프 생성에 사용할 텍스트 파일 경로")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🌐 LLM 기반 지식 그래프 생성기")
    print("=" * 60)
    
    # 그래프 생성기 인스턴스 생성
    generator = GraphGenerator(
        server_url=args.server,
        llm_model=args.llm,
        embedding_model=args.embedding
    )
    
    print(f"\n✅ 시스템 준비 완료!")
    
    # 기존 그래프 보강 모드
    if args.enrich:
        generator.enrich_knowledge_graph(
            graph_file=args.enrich,
            output_file=args.output
        )
    # 새 그래프 생성 모드 (텍스트 파일 또는 도메인 기반)
    else:
        # 텍스트 파일이 지정되지 않았고 기본 텍스트 파일이 있는 경우 자동으로 사용
        if not args.text and os.path.exists("context.txt"):
            args.text = "context.txt"
            print(f"📄 기본 텍스트 파일 'context.txt'를 사용합니다.")
        
        generator.create_knowledge_graph(
            domain=args.domain,
            text_file=args.text,
            num_nodes=args.nodes,
            output_file=args.output
        )

if __name__ == "__main__":
    main()