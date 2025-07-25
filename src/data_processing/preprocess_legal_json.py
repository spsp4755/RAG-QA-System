import os
import json
from glob import glob

def extract_chunks_from_json(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    doc = data["document"]
    doc_title = doc.get("title", "")
    doc_type = doc.get("doc_type", "")
    created_time = doc.get("created_time", "")
    chunks = []
    for sub in doc.get("sub_documents", []):
        # 여러 contents가 있으면 합치기
        text = " ".join([c.get("text", "") for c in sub.get("contents", []) if c.get("text")])
        if not text.strip():
            continue
        chunk = {
            "chunk_id": f"{os.path.basename(json_path)}_{sub['id']}",
            "text": text,
            "metadata": safe_metadata({
                "doc_type": doc_type,
                "title": doc_title,
                "article": sub.get("article"),
                "sort_order": sub.get("sort_order"),
                "created_time": created_time,
                "content_labels": ",".join([str(label) for label in sub.get("content_labels", []) if label is not None])
            })
        }
        chunks.append(chunk)
    return chunks

def safe_metadata(d):
    return {k: ("" if v is None else v) for k, v in d.items()}

def process_all_jsons(input_dir, output_path):
    all_chunks = []
    for json_path in glob(f"{input_dir}/**/*.json", recursive=True):
        all_chunks.extend(extract_chunks_from_json(json_path))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_chunks)} chunks to {output_path}")

if __name__ == "__main__":
    # 경로는 필요에 따라 수정
    input_dir = "data/raw/contract_legal_documents"
    output_path = "data/processed/contract_legal_chunks.json"
    process_all_jsons(input_dir, output_path)