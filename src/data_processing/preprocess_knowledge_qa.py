import os
import json
import glob

def extract_qa_from_labeling(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    taskinfo = data['taskinfo']
    info = data['info']
    # sentences 리스트를 하나의 문자열로 합침
    context = " ".join(taskinfo.get('sentences', []))
    qa = {
        "question": taskinfo['input'],
        "answer": taskinfo['output'],
        "context": context,
        "instruction": taskinfo.get('instruction', ''),
        "metadata": {
            "doc_id": info.get("doc_id", ""),
            "casenames": info.get("casenames", ""),
            "court": info.get("normalized_court", ""),
            "date": info.get("announce_date", ""),
            "casetype": info.get("casetype", ""),
        }
    }
    return qa

def process_all_labeling(input_dirs, output_path):
    all_qa = []
    for input_dir in input_dirs:
        for file in glob.glob(os.path.join(input_dir, "*.json")):
            qa = extract_qa_from_labeling(file)
            all_qa.append(qa)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_qa, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_dirs = [
        "data/raw/knowledge_data/Training/02.라벨링데이터",
        "data/raw/knowledge_data/Validation/02.라벨링데이터"
    ]
    output_path = "data/processed/knowledge_qa.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    process_all_labeling(input_dirs, output_path) 