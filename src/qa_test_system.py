import json
import time
import os
import argparse
import requests
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from datetime import datetime
import networkx as nx

class QATestSystem:
    """4가지 방식으로 질의응답을 테스트하는 시스템"""
    
    def __init__(self, server_url, model_name="gemma3:27b",
                 onnx_model_path="model2.onnx", db_path="rag_db.json", 
                 graph_path="knowledge_graph.json", text_path="context.txt"):
        
        self.server_url = server_url
        self.model_name = model_name
        self.onnx_model_path = onnx_model_path
        self.db_path = db_path
        self.graph_path = graph_path
        self.text_path = text_path
        
        print(f"🚀 QA 테스트 시스템 초기화 중...")
        print(f"   LLM 서버: {server_url}")
        print(f"   LLM 모델: {model_name}")
        
        # ONNX 모델 및 토크나이저 로드 시도
        try:
            print(f"   ONNX 모델 로드 중: {onnx_model_path}")
            self.ort_session = ort.InferenceSession(onnx_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
            print("   ONNX 모델 로드 완료")
            self.embedding_available = True
        except Exception as e:
            print(f"⚠️ ONNX 모델 로드 실패: {e}")
            print("   RAG 및 GraphRAG 방식은 임베딩 없이 진행됩니다.")
            self.embedding_available = False
            self.ort_session = None
            self.tokenizer = None
        
        # 텍스트 파일 로드 시도
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                self.context_text = f.read()
            print(f"   컨텍스트 파일 로드 완료: {text_path} ({len(self.context_text)} 자)")
            self.context_available = True
        except Exception as e:
            print(f"⚠️ 컨텍스트 파일 로드 실패: {e}")
            print("   컨텍스트 기반 방식은 건너뛰게 됩니다.")
            self.context_available = False
            self.context_text = ""
        
        # DB 파일 로드 시도 (RAG용)
        try:
            if os.path.exists(db_path):
                print(f"   RAG DB 로드 중: {db_path}")
                with open(db_path, 'r', encoding='utf-8') as f:
                    self.db_data = json.load(f)
                print(f"   RAG DB 로드 완료 ({len(self.db_data['chunks'])} 청크)")
                self.rag_available = True
            else:
                print(f"⚠️ RAG DB 파일을 찾을 수 없음: {db_path}")
                self.rag_available = False
                self.db_data = None
        except Exception as e:
            print(f"⚠️ RAG DB 로드 실패: {e}")
            self.rag_available = False
            self.db_data = None
        
        # 지식 그래프 로드 시도 (GraphRAG용)
        try:
            if os.path.exists(graph_path):
                print(f"   지식 그래프 로드 중: {graph_path}")
                self.load_graph(graph_path)
                print(f"   지식 그래프 로드 완료 ({len(self.G.nodes)} 노드, {len(self.G.edges)} 엣지)")
                self.graph_available = True
            else:
                print(f"⚠️ 지식 그래프 파일을 찾을 수 없음: {graph_path}")
                self.graph_available = False
        except Exception as e:
            print(f"⚠️ 지식 그래프 로드 실패: {e}")
            self.graph_available = False
            self.G = None
            self.node_info = None
            self.node_embeddings = None
        
        print("✅ 초기화 완료")
    
    def load_graph(self, graph_path):
        """지식 그래프 로드"""
        with open(graph_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        # 그래프 생성
        self.G = nx.DiGraph()
        
        # 노드 정보 및 임베딩 저장용 딕셔너리
        self.node_info = {}
        self.node_embeddings = {}
        
        # 노드 추가
        for node in graph_data.get("nodes", []):
            node_id = node.get("id")
            self.G.add_node(node_id)
            
            # 노드 정보 저장
            self.node_info[node_id] = {
                "name": node.get("name", ""),
                "type": node.get("type", ""),
                "description": node.get("description", ""),
                "properties": node.get("properties", {})
            }
            
            # 임베딩이 있으면 저장
            if "embedding" in node:
                self.node_embeddings[node_id] = np.array(node["embedding"])
        
        # 엣지 추가
        for edge in graph_data.get("edges", []):
            source = edge.get("source")
            target = edge.get("target")
            relation = edge.get("relation", "관련")
            
            if source in self.G and target in self.G:
                self.G.add_edge(source, target, relation=relation)
    
    def call_llm(self, prompt):
        """LLM API 호출하여 응답 받기"""
        try:
            response = requests.post(f"{self.server_url}/api/generate", json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.1,
                "max_tokens": 2048
            })
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return f"API 호출 오류: 상태 코드 {response.status_code}"
        except Exception as e:
            return f"API 호출 중 예외 발생: {str(e)}"
    
    def method1_direct_query(self, question):
        """방식 1: 직접 질문"""
        print(f"   방식 1: 직접 질문 처리 중...")
        
        # 프롬프트 생성
        prompt_start_time = time.time()
        prompt = f"다음 질문에 명확하게 답해주세요:\n\n질문: {question}\n\n답변:"
        prompt_time = time.time() - prompt_start_time
        
        # 응답 생성
        generation_start_time = time.time()
        response = self.call_llm(prompt)
        generation_time = time.time() - generation_start_time
        
        # 총 시간
        total_time = prompt_time + generation_time
        
        return {
            "response": response,
            "prompt_time": prompt_time,
            "generation_time": generation_time,
            "total_time": total_time,
            "prompt": prompt
        }
    
    def method2_context_query(self, question):
        """방식 2: 컨텍스트 기반 질문"""
        print(f"   방식 2: 컨텍스트 기반 질문 처리 중...")
        
        if not self.context_available:
            return {
                "response": "컨텍스트 파일을 사용할 수 없어 응답할 수 없습니다.",
                "prompt_time": 0,
                "generation_time": 0,
                "total_time": 0,
                "prompt": "컨텍스트 파일 없음"
            }
        
        # 프롬프트 생성
        prompt_start_time = time.time()
        prompt = f"""다음 컨텍스트 정보를 바탕으로 질문에 답변해주세요:

[컨텍스트]
{self.context_text}

[질문]
{question}

[답변]"""
        prompt_time = time.time() - prompt_start_time
        
        # 응답 생성
        generation_start_time = time.time()
        response = self.call_llm(prompt)
        generation_time = time.time() - generation_start_time
        
        # 총 시간
        total_time = prompt_time + generation_time
        
        return {
            "response": response,
            "prompt_time": prompt_time,
            "generation_time": generation_time,
            "total_time": total_time,
            "prompt": prompt
        }
    
    def get_query_embedding(self, query):
        """쿼리 텍스트를 임베딩 벡터로 변환"""
        # E5 모델의 쿼리 형식 적용
        query_text = f"query: {query}"
        
        # 토큰화
        encoded_input = self.tokenizer(
            query_text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='np'
        )
        
        # token_type_ids 추가 (ONNX 모델 필요)
        if 'token_type_ids' not in encoded_input:
            batch_size = encoded_input['input_ids'].shape[0]
            seq_length = encoded_input['input_ids'].shape[1]
            encoded_input['token_type_ids'] = np.zeros((batch_size, seq_length), dtype=np.int64)
        
        # ONNX 모델로 추론
        model_inputs = {
            'input_ids': encoded_input['input_ids'],
            'attention_mask': encoded_input['attention_mask'],
            'token_type_ids': encoded_input['token_type_ids']
        }
        
        # 모델 실행
        outputs = self.ort_session.run(None, model_inputs)
        
        # 어텐션 마스크를 사용한 평균 풀링 (E5 표준 방식)
        attention_mask = encoded_input['attention_mask']
        embedding_output = outputs[0]
        
        # 마스크된 평균 계산 (배치 크기 1)
        attention_mask = attention_mask.reshape(attention_mask.shape[0], attention_mask.shape[1], 1)
        sum_embeddings = np.sum(embedding_output * attention_mask, axis=1)
        sum_mask = np.sum(attention_mask, axis=1)
        pooled_embedding = sum_embeddings / sum_mask
        
        # 배치 차원 제거 (배치 크기 1)
        embedding = pooled_embedding.squeeze(0)
        
        # 정규화
        norm = np.linalg.norm(embedding)
        normalized_embedding = embedding / norm
        
        return normalized_embedding
    
    def cosine_similarity(self, a, b):
        """두 벡터 간의 코사인 유사도 계산"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def search_similar(self, query, k=3):
        """벡터 유사도 검색으로 관련 문서 찾기"""
        # 쿼리 임베딩
        query_embedding = self.get_query_embedding(query)
        
        # 데이터베이스에서 데이터 가져오기
        db_embeddings = np.array(self.db_data["embeddings"])
        chunks = self.db_data["chunks"]
        metadatas = self.db_data["metadatas"]
        
        # 코사인 유사도 계산
        similarities = []
        for i, doc_embedding in enumerate(db_embeddings):
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, similarity))
        
        # 유사도 기준 내림차순 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 k개 결과 반환
        results = []
        for idx, sim in similarities[:k]:
            results.append({
                "text": chunks[idx],
                "metadata": metadatas[idx],
                "similarity": sim
            })
        
        return results
    
    def format_rag_prompt(self, contexts, query):
        """검색 결과와 질문으로 프롬프트 생성"""
        context_text = "\n\n".join([f"[출처: {ctx['metadata']['source']}]\n{ctx['text']}" 
                                  for ctx in contexts])
        
        return f"""다음 정보를 바탕으로 질문에 답변해주세요:

[참고 정보]
{context_text}

[질문]
{query}

[답변]"""
    
    def method3_rag_query(self, question, k=3):
        """방식 3: RAG 기반 질문"""
        print(f"   방식 3: RAG 기반 질문 처리 중...")
        
        if not self.rag_available or not self.embedding_available:
            return {
                "response": "RAG 데이터베이스 또는 임베딩 모델을 사용할 수 없어 응답할 수 없습니다.",
                "prompt_time": 0,
                "generation_time": 0,
                "total_time": 0,
                "prompt": "RAG 불가능",
                "contexts": []
            }
        
        prompt_start_time = time.time()
        
        # 관련 문서 검색
        contexts = self.search_similar(question, k)
        
        # 프롬프트 생성
        prompt = self.format_rag_prompt(contexts, question)
        prompt_time = time.time() - prompt_start_time
        
        # 응답 생성
        generation_start_time = time.time()
        response = self.call_llm(prompt)
        generation_time = time.time() - generation_start_time
        
        # 총 시간
        total_time = prompt_time + generation_time
        
        return {
            "response": response,
            "prompt_time": prompt_time,
            "generation_time": generation_time,
            "total_time": total_time,
            "prompt": prompt,
            "contexts": contexts
        }
    
    def search_graph(self, query, top_k=10):
        """지식 그래프에서 검색"""
        print(f"🔍 그래프 검색 시작: '{query}' (top_k={top_k})")
        
        start_time = time.time()
        results = []
        
        # 쿼리 임베딩 계산
        query_embedding = None
        if self.embedding_available and self.ort_session is not None:
            try:
                query_embedding = self.get_query_embedding(query)
            except Exception as e:
                print(f"⚠️ 쿼리 임베딩 계산 실패: {e}")
        
        # 임베딩 유사도 검색
        embedding_results = []
        if query_embedding is not None:
            print("   임베딩 기반 검색 수행 중...")
            
            # 모든 노드와의 유사도 계산
            for node_id, node_info in self.node_info.items():
                node_embedding = self.get_node_embedding(node_id)
                if node_embedding is not None:
                    sim = self.cosine_similarity(query_embedding, node_embedding)
                    embedding_results.append((node_id, sim, "임베딩 유사도"))
            
            # 상위 결과 선택
            embedding_results.sort(key=lambda x: x[1], reverse=True)
            embedding_results = embedding_results[:top_k*2]  # 충분히 많은 후보 선택
            print(f"   임베딩 검색 결과: {len(embedding_results)}개 노드 발견")
        
        # 텍스트 매칭 검색
        keyword_results = []
        if len(embedding_results) < top_k:
            print("   텍스트 매칭 검색 수행 중...")
            
            query_tokens = set(query.lower().split())
            for node_id, node_info in self.node_info.items():
                # 노드 텍스트 추출
                node_text = " ".join([
                    node_info.get("name", ""),
                    node_info.get("description", ""),
                    str(node_info.get("properties", {}))
                ]).lower()
                
                # 토큰 매칭 점수
                matches = sum(1 for token in query_tokens if token in node_text)
                if matches > 0:
                    score = matches / len(query_tokens)
                    keyword_results.append((node_id, score, "텍스트 매칭"))
            
            # 상위 결과 선택
            keyword_results.sort(key=lambda x: x[1], reverse=True)
            keyword_results = keyword_results[:top_k]
            print(f"   텍스트 매칭 결과: {len(keyword_results)}개 노드 발견")
        
        # 결과 합치기
        all_results = embedding_results + keyword_results
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        # 중복 제거 (높은 점수 유지)
        seen_nodes = set()
        filtered_results = []
        for node_id, score, method in all_results:
            if node_id not in seen_nodes:
                seen_nodes.add(node_id)
                filtered_results.append((node_id, score, method))
        
        # 그래프 기반 확장 (관련 노드 포함)
        final_results = list(filtered_results)
        if len(filtered_results) > 0:
            print("   그래프 기반 확장 수행 중...")
            expansion_nodes = set()
            
            # 최상위 노드 주변 그래프 탐색
            for node_id, _, _ in filtered_results[:min(3, len(filtered_results))]:
                # 들어오는 엣지
                for edge in self.G.predecessors(node_id):
                    pred = edge
                    if pred and pred not in seen_nodes:
                        expansion_nodes.add((pred, 0.7, "그래프 연결"))
                
                # 나가는 엣지
                for edge in self.G.successors(node_id):
                    succ = edge
                    if succ and succ not in seen_nodes:
                        expansion_nodes.add((succ, 0.6, "그래프 연결"))
            
            # 확장된 노드 추가
            for node_id, score, method in expansion_nodes:
                final_results.append((node_id, score, method))
                seen_nodes.add(node_id)
        
        # 최종 결과 정렬 및 상위 선택
        final_results.sort(key=lambda x: x[1], reverse=True)
        final_results = final_results[:top_k]
        
        # 결과 정보 구성
        for node_id, score, method in final_results:
            node_info = self.node_info.get(node_id, {})
            
            # 엣지 정보 수집
            incoming_edges = []
            for pred in self.G.predecessors(node_id):
                edge_data = self.G.get_edge_data(pred, node_id)
                pred_info = self.node_info.get(pred, {})
                incoming_edges.append({
                    "source_id": pred,
                    "source_name": pred_info.get("name", "알 수 없음"),
                    "relation": edge_data.get("relation", "관련")
                })
            
            outgoing_edges = []
            for succ in self.G.successors(node_id):
                edge_data = self.G.get_edge_data(node_id, succ)
                succ_info = self.node_info.get(succ, {})
                outgoing_edges.append({
                    "target_id": succ,
                    "target_name": succ_info.get("name", "알 수 없음"),
                    "relation": edge_data.get("relation", "관련")
                })
            
            # 결과 추가
            results.append({
                "id": node_id,
                "name": node_info.get("name", ""),
                "type": node_info.get("type", ""),
                "description": node_info.get("description", ""),
                "properties": node_info.get("properties", {}),
                "score": score,
                "method": method,
                "incoming_edges": incoming_edges,
                "outgoing_edges": outgoing_edges
            })
        
        search_time = time.time() - start_time
        print(f"✅ 그래프 검색 완료: {len(results)}개 노드 ({search_time:.2f}초)")
        
        return results
    
    def get_node_embedding(self, node_id):
        """노드 임베딩 가져오기 (캐시 또는 계산)"""
        if not self.embedding_available or self.ort_session is None:
            return None
        
        # 캐시에 있으면 재사용
        if node_id in self.node_embeddings:
            return self.node_embeddings[node_id]
        
        # 노드 텍스트 준비
        node_info = self.node_info.get(node_id, {})
        node_text = " ".join([
            node_info.get("name", ""),
            node_info.get("description", ""),
            str(node_info.get("properties", {}))
        ])
        
        # 임베딩 계산
        try:
            embedding = self.compute_embedding(node_text)
            if embedding is not None:
                self.node_embeddings[node_id] = embedding
            return embedding
        except Exception as e:
            print(f"⚠️ 노드 임베딩 계산 실패({node_id}): {e}")
            return None
    
    def compute_embedding(self, text):
        """텍스트에 대한 임베딩 계산"""
        if not self.embedding_available or self.ort_session is None:
            return None
        
        try:
            # 토크나이징
            inputs = self.tokenizer(text, padding=True, truncation=True, 
                                   return_tensors="np", max_length=512)
            
            # ONNX 모델 실행을 위한 입력 준비
            ort_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64)
            }
            
            # token_type_ids가 필요하지만 토크나이저에서 제공하지 않는 경우 직접 생성
            if "token_type_ids" not in inputs and "token_type_ids" in [input.name for input in self.ort_session.get_inputs()]:
                token_type_ids = np.zeros_like(inputs["input_ids"])
                ort_inputs["token_type_ids"] = token_type_ids
            # 토크나이저에서 제공하는 경우
            elif "token_type_ids" in inputs:
                ort_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)
            
            # ONNX 실행
            embeddings = self.ort_session.run(None, ort_inputs)[0]
            
            # 마스킹된 토큰의 평균 임베딩 계산 (CLS 토큰 또는 평균)
            return embeddings[0, 0]  # CLS 토큰 사용
        except Exception as e:
            print(f"⚠️ 임베딩 계산 실패: {e}")
            return None
    
    def build_graph_prompt(self, query, search_results):
        """검색 결과로부터 그래프 기반 프롬프트 구성"""
        prompt = f"""### 질문:
{query}

### 관련 정보:
"""
        
        # 노드 유형별 그룹화
        node_groups = {}
        for result in search_results:
            node_type = result.get("type", "기타")
            if node_type not in node_groups:
                node_groups[node_type] = []
            node_groups[node_type].append(result)
        
        # 유형별로 노드 정보 추가
        for node_type, nodes in node_groups.items():
            prompt += f"\n## {node_type} 정보:\n"
            
            for node in nodes:
                prompt += f"\n### {node.get('name', '이름 없음')}\n"
                
                # 설명 추가
                if node.get("description"):
                    prompt += f"{node['description']}\n"
                
                # 속성 추가
                if node.get("properties"):
                    prompt += "\n속성:\n"
                    for key, value in node["properties"].items():
                        prompt += f"- {key}: {value}\n"
                
                # 관계 정보 추가
                if node.get("incoming_edges"):
                    prompt += "\n들어오는 관계:\n"
                    for edge in node["incoming_edges"]:
                        prompt += f"- {edge['source_name']} → {edge['relation']} → {node['name']}\n"
                
                if node.get("outgoing_edges"):
                    prompt += "\n나가는 관계:\n"
                    for edge in node["outgoing_edges"]:
                        prompt += f"- {node['name']} → {edge['relation']} → {edge['target_name']}\n"
        
        # LLM 지시 추가
        prompt += f"""
### 지시사항:
- 위 정보를 바탕으로 질문에 정확하게 답변해주세요.
- 정보가 부족하면 솔직히 모른다고 말해주세요.
- 답변은 명확하고 구체적으로 작성해주세요.
- 가능하면 제공된 정보의 출처를 언급해주세요.

### 답변:
"""
        
        return prompt
    
    def method4_graph_rag_query(self, question, top_k=10):
        """방식 4: GraphRAG 기반 질문"""
        print(f"   방식 4: GraphRAG 기반 질문 처리 중...")
        
        if not self.graph_available:
            return {
                "response": "지식 그래프를 사용할 수 없어 응답할 수 없습니다.",
                "prompt_time": 0,
                "generation_time": 0,
                "total_time": 0,
                "prompt": "GraphRAG 불가능",
                "search_results": []
            }
        
        prompt_start_time = time.time()
        
        # 그래프에서 관련 노드 검색
        search_results = self.search_graph(question, top_k)
        
        # 프롬프트 구성
        prompt = self.build_graph_prompt(question, search_results)
        prompt_time = time.time() - prompt_start_time
        
        # 응답 생성
        generation_start_time = time.time()
        response = self.call_llm(prompt)
        generation_time = time.time() - generation_start_time
        
        # 총 시간
        total_time = prompt_time + generation_time
        
        return {
            "response": response,
            "prompt_time": prompt_time,
            "generation_time": generation_time,
            "total_time": total_time,
            "prompt": prompt,
            "search_results": search_results
        }
    
    def process_question(self, question):
        """4가지 방식으로 질문 처리"""
        print(f"\n📝 질문 처리: '{question}'")
        
        results = {
            "question": question,
            "timestamp": datetime.now().isoformat(),
            "methods": {}
        }
        
        # 방식 1: 직접 질문
        method1_result = self.method1_direct_query(question)
        results["methods"]["direct_query"] = method1_result
        print(f"   방식 1 완료: {method1_result['total_time']:.2f}초")
        
        # 방식 2: 컨텍스트 기반 질문
        method2_result = self.method2_context_query(question)
        results["methods"]["context_query"] = method2_result
        print(f"   방식 2 완료: {method2_result['total_time']:.2f}초")
        
        # 방식 3: RAG 기반 질문
        method3_result = self.method3_rag_query(question)
        results["methods"]["rag_query"] = method3_result
        print(f"   방식 3 완료: {method3_result['total_time']:.2f}초")
        
        # 방식 4: GraphRAG 기반 질문
        method4_result = self.method4_graph_rag_query(question)
        results["methods"]["graph_rag_query"] = method4_result
        print(f"   방식 4 완료: {method4_result['total_time']:.2f}초")
        
        return results

def main():
    """메인 함수"""
    # 명령줄 인터페이스 설정
    parser = argparse.ArgumentParser(description="QA 시스템 테스트 도구")
    parser.add_argument("--qa_file", default="qa_dataset_1_300.json", help="QA 데이터셋 파일 경로")
    parser.add_argument("--output", default="qa_test_results.json", help="결과 저장 파일 경로")
    parser.add_argument("--server", help="LLM 서버 URL")
    parser.add_argument("--model", default="gemma3:27b", help="LLM 모델 이름")
    parser.add_argument("--onnx_model", default="model2.onnx", help="ONNX 모델 파일 경로")
    parser.add_argument("--db", default="rag_db.json", help="RAG 데이터베이스 경로")
    parser.add_argument("--graph", default="knowledge_graph.json", help="지식 그래프 파일 경로")
    parser.add_argument("--text", default="context.txt", help="컨텍스트 텍스트 파일 경로")
    parser.add_argument("--limit", type=int, default=10, help="처리할 질문 수 제한 (0은 모두 처리)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 QA 테스트 시스템")
    print("=" * 60)
    
    # QA 데이터셋 로드
    try:
        with open(args.qa_file, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        print(f"✅ QA 데이터셋 로드 완료: {args.qa_file} ({len(qa_data)} 개 질문)")
    except Exception as e:
        print(f"❌ QA 데이터셋 로드 실패: {e}")
        return
    
    # 테스트 시스템 초기화
    qa_system = QATestSystem(
        server_url=args.server,
        model_name=args.model,
        onnx_model_path=args.onnx_model,
        db_path=args.db,
        graph_path=args.graph,
        text_path=args.text
    )
    
    # 질문 처리
    all_results = []
    question_count = min(args.limit, len(qa_data)) if args.limit > 0 else len(qa_data)
    
    print(f"\n📊 {question_count}개 질문 처리 시작...")
    
    for i, qa_item in enumerate(qa_data[:question_count], 1):
        question = qa_item["question"]
        print(f"\n[{i}/{question_count}] 질문: '{question}'")
        
        # 질문 처리
        result = qa_system.process_question(question)
        all_results.append(result)
        
        # 진행 상황 표시
        print(f"✅ 질문 {i}/{question_count} 완료")
        
        # 결과 저장 (중간 저장)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        # 마지막 질문이 아니면 잠시 대기
        if i < question_count:
            time.sleep(1)
    
    # 최종 결과 요약
    print("\n📋 테스트 결과 요약:")
    
    # 각 방식별 평균 시간 계산
    method_times = {
        "direct_query": {"prompt": 0, "generation": 0, "total": 0},
        "context_query": {"prompt": 0, "generation": 0, "total": 0},
        "rag_query": {"prompt": 0, "generation": 0, "total": 0},
        "graph_rag_query": {"prompt": 0, "generation": 0, "total": 0}
    }
    
    for result in all_results:
        for method_name, method_data in result["methods"].items():
            method_times[method_name]["prompt"] += method_data["prompt_time"]
            method_times[method_name]["generation"] += method_data["generation_time"]
            method_times[method_name]["total"] += method_data["total_time"]
    
    for method_name, times in method_times.items():
        avg_prompt = times["prompt"] / len(all_results)
        avg_generation = times["generation"] / len(all_results)
        avg_total = times["total"] / len(all_results)
        
        print(f"\n방식: {method_name}")
        print(f"  - 평균 프롬프트 생성 시간: {avg_prompt:.4f}초")
        print(f"  - 평균 응답 생성 시간: {avg_generation:.4f}초")
        print(f"  - 평균 총 시간: {avg_total:.4f}초")
    
    print(f"\n🎉 테스트 완료. 결과가 저장되었습니다: {args.output}")

if __name__ == "__main__":
    main() 