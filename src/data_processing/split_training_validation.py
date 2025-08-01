import os
import json
import glob
from src.data_processing.preprocess_knowledge_qa import extract_qa_from_labeling

def process_training_data():
    """Training 데이터만 처리"""
    print("📚 Training 데이터 처리 중...")
    training_qa = []
    
    training_dir = "data/raw/knowledge_data/Training/02.라벨링데이터"
    for file in glob.glob(os.path.join(training_dir, "*.json")):
        qa = extract_qa_from_labeling(file)
        training_qa.append(qa)
    
    output_path = "data/processed/training_qa.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_qa, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Training 데이터 저장 완료: {len(training_qa)}개")
    return output_path

def process_validation_data():
    """Validation 데이터만 처리"""
    print("🧪 Validation 데이터 처리 중...")
    validation_qa = []
    
    validation_dir = "data/raw/knowledge_data/Validation/02.라벨링데이터"
    for file in glob.glob(os.path.join(validation_dir, "*.json")):
        qa = extract_qa_from_labeling(file)
        validation_qa.append(qa)
    
    output_path = "data/processed/validation_qa.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(validation_qa, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Validation 데이터 저장 완료: {len(validation_qa)}개")
    return output_path

def main():
    print("🔄 Training/Validation 데이터 분리 시작...")
    
    # Training 데이터 처리
    training_path = process_training_data()
    
    # Validation 데이터 처리
    validation_path = process_validation_data()
    
    print("\n📊 데이터 분리 완료!")
    print(f"Training: {training_path}")
    print(f"Validation: {validation_path}")
    
    # 데이터 통계 출력
    with open(training_path, 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    
    with open(validation_path, 'r', encoding='utf-8') as f:
        validation_data = json.load(f)
    
    print(f"\n📈 데이터 통계:")
    print(f"Training 데이터: {len(training_data)}개")
    print(f"Validation 데이터: {len(validation_data)}개")
    print(f"총 데이터: {len(training_data) + len(validation_data)}개")

if __name__ == "__main__":
    main() 