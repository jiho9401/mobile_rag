from sentence_transformers import SentenceTransformer
import torch
import os

def convert_model_to_onnx(model_name="intfloat/multilingual-e5-small", output_path="model2.onnx"):
    """Sentence Transformer 모델을 ONNX 형식으로 변환"""
    print(f"모델 로딩 중: {model_name}")
    model = SentenceTransformer(model_name)
    
    # 강제로 CPU 모드로 설정 (디바이스 일관성 유지)
    model = model.to('cpu')
    
    # 더미 입력 준비 (CPU 텐서로 생성)
    dummy_input_ids = torch.ones((1, 128), dtype=torch.long, device='cpu')
    dummy_attention_mask = torch.ones((1, 128), dtype=torch.long, device='cpu')
    dummy_token_type_ids = torch.zeros((1, 128), dtype=torch.long, device='cpu')
    
    # PyTorch 모델 가져오기
    pytorch_model = model._modules['0'].auto_model
    pytorch_model = pytorch_model.to('cpu')  # 명시적으로 CPU로 이동
    pytorch_model.eval()  # 평가 모드로 설정
    
    # ONNX 변환
    print(f"ONNX 모델로 변환 중...")
    torch.onnx.export(
        pytorch_model,
        (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids),
        output_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["output"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size"}
        },
        opset_version=14,
        do_constant_folding=True,  # 상수 폴딩 최적화 적용
        verbose=False
    )
    
    print(f"✅ 변환 완료: {output_path}")
    print(f"  - 모델 크기: {os.path.getsize(output_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    convert_model_to_onnx() 