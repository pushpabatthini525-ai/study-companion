import json
from ingest_video import make_chunks

url = input("YouTube link: ")
chunks = make_chunks(url)

with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print("Saved", len(chunks), "chunks to chunks.json")