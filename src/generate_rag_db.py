import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import os
import json
import gc
import argparse

def generate_db(input_paths=None, output_path="rag_db.json", chunk_size=200, chunk_overlap=20, model_path="model2.onnx"):
    """문서를 청크로 나누고 ONNX 모델로 임베딩하여 벡터 데이터베이스 생성"""
    if input_paths is None:
        input_paths = ["context.txt"]
    
    # ONNX 모델 및 토크나이저 로드
    print("ONNX 모델 로딩 중...")
    ort_session = ort.InferenceSession(model_path)
    # E5 모델용 토크나이저로 변경
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    
    all_chunks = []
    all_metadatas = []
    total_files = len(input_paths)
    
    def split_text(text, size=chunk_size, overlap=chunk_overlap):
        """텍스트를 청크로 분할"""
        chunks = []
        for i in range(0, len(text), size - overlap):
            chunk = text[i:i + size]
            if chunk:  # 빈 청크는 건너뜀
                chunks.append(chunk)
        return chunks
    
    # ONNX 모델을 사용하여 임베딩 생성
    def get_embeddings(texts, batch_size=8):  # 모바일 장치에 맞게 작은 배치 크기
        """문서 텍스트에서 임베딩 생성 (E5 모델용)"""
        all_embeddings = []
        total_texts = len(texts)
        
        for i in range(0, total_texts, batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # 진행상황 표시
            if i % (batch_size * 4) == 0:
                print(f"  - 임베딩 진행: {i}/{total_texts} ({i/total_texts*100:.1f}%)")
            
            # E5 모델의 패시지 형식 적용
            formatted_texts = [f"passage: {text}" for text in batch_texts]
            
            # 토큰화
            encoded_input = tokenizer(
                formatted_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='np'
            )
            
            # token_type_ids 추가 (ONNX 모델 필요)
            # E5 모델은 실제로 이를 사용하지 않지만 ONNX 모델 입력으로 필요
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
            outputs = ort_session.run(None, model_inputs)
            
            # 어텐션 마스크를 사용한 평균 풀링 (E5 표준 방식)
            attention_mask = encoded_input['attention_mask'].reshape(
                encoded_input['attention_mask'].shape[0], 
                encoded_input['attention_mask'].shape[1], 
                1
            )
            embedding_output = outputs[0]
            
            # 마스크된 평균 계산 (토큰 차원을 따라 합산)
            sum_embeddings = np.sum(embedding_output * attention_mask, axis=1)
            sum_mask = np.sum(attention_mask, axis=1)
            pooled_embeddings = sum_embeddings / sum_mask
            
            # 정규화
            norms = np.linalg.norm(pooled_embeddings, axis=1, keepdims=True)
            normalized_embeddings = pooled_embeddings / norms
            
            all_embeddings.append(normalized_embeddings)
            
            # 배치 처리 후 메모리 명시적 정리 (모바일 장치 최적화)
            if i % (batch_size * 8) == 0 and i > 0:
                gc.collect()
        
        # 모든 임베딩 결합
        return np.vstack(all_embeddings)
    
    # 각 파일 처리
    for file_idx, input_path in enumerate(input_paths, 1):
        try:
            print(f"[{file_idx}/{total_files}] 파일 처리 중: {input_path}")
            
            if not os.path.exists(input_path):
                print(f"파일을 찾을 수 없음: {input_path}")
                continue
                
            with open(input_path, "r", encoding="utf-8") as f:
                full_text = f.read()
            
            # 텍스트 분할
            chunks = split_text(full_text)
            print(f"  - 분할된 청크 수: {len(chunks)}")
            
            # 메타데이터 추가
            metadatas = [{"source": input_path} for _ in range(len(chunks))]
            
            all_chunks.extend(chunks)
            all_metadatas.extend(metadatas)
            
        except Exception as e:
            print(f"⚠️ {input_path} 처리 중 오류: {e}")
    
    # 임베딩 생성 (배치 처리)
    print(f"임베딩 생성 중 (청크 수: {len(all_chunks)})...")
    embeddings = get_embeddings(all_chunks)
    
    # 저장 디렉토리 확인 및 생성
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 벡터와 텍스트 저장
    print(f"벡터 데이터베이스 저장 중: {output_path}")
    database = {
        "chunks": all_chunks,
        "embeddings": embeddings.tolist(),  # numpy 배열을 리스트로 변환
        "metadatas": all_metadatas
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(database, f, ensure_ascii=False)
    
    print(f"✅ RAG 데이터베이스 생성 완료")
    print(f"  - 데이터베이스: {output_path}")
    print(f"  - 청크 수: {len(all_chunks)}")
    print(f"  - 임베딩 차원: {embeddings.shape[1]}")
    
    # 메모리 정리
    del embeddings
    gc.collect()

def main():
    parser = argparse.ArgumentParser(description='문서를 ONNX 모델로 임베딩하여 RAG 데이터베이스 생성')
    parser.add_argument('--input', '-i', nargs='+', help='입력 파일 경로 (여러 개 가능)')
    parser.add_argument('--output', '-o', default='rag_db.json', help='출력 파일 경로')
    parser.add_argument('--chunk_size', '-c', type=int, default=200, help='텍스트 청크 크기')
    parser.add_argument('--chunk_overlap', '-v', type=int, default=20, help='청크 오버랩 크기')
    parser.add_argument('--model', '-m', default='model2.onnx', help='ONNX 모델 파일 경로')
    
    args = parser.parse_args()
    
    generate_db(
        input_paths=args.input, 
        output_path=args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        model_path=args.model
    )

if __name__ == "__main__":
    main()
