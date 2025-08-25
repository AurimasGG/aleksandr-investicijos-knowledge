import zipfile
import os
import re
import json
import pickle
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from zipfile import ZipFile, ZIP_DEFLATED

# Įvesties failas (tavo turimas originalus zip su transkriptais)
ZIP_PATH = "aleksandr - youutbe - 1-89.zip"

# Laikini aplankai
EXTRACT_DIR = "aleksandr_youtube_transcripts"
OUT_DIR = "aleksandr_knowledge_pack"
ZIP_OUT = "aleksandr_knowledge_pack.zip"

# 1) Išarchyvuojam visus transkriptus
Path(EXTRACT_DIR).mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    z.extractall(EXTRACT_DIR)

# 2) Nuskaitymas + valymas
def read_file(path):
    try:
        if path.suffix.lower() in [".txt", ".md"]:
            return path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(data, dict):
                if "segments" in data:
                    return "\n".join(seg.get("text", "") for seg in data["segments"])
                if "items" in data:
                    return "\n".join(it.get("text", "") for it in data["items"])
                if "text" in data and isinstance(data["text"], str):
                    return data["text"]
            if isinstance(data, list):
                return "\n".join(seg.get("text", str(seg)) for seg in data)
    except:
        return ""
    return ""

def clean_text(t):
    if not t:
        return ""
    t = re.sub(r"\[?\b\d{1,2}:\d{2}(?::\d{2})?\]?", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

files = []
for root, dirs, filenames in os.walk(EXTRACT_DIR):
    for fname in filenames:
        p = Path(root) / fname
        if p.suffix.lower() in [".txt", ".md", ".json"]:
            files.append(p)

records = []
for p in sorted(files):
    raw = read_file(p)
    text = clean_text(raw)
    if not text:
        continue
    pieces = re.split(r"(?<=[\.\!\?])\s+", text)
    chunk, chunk_chars, chunk_id = [], 0, 0
    MAX_CHARS = 1500
    for sent in pieces:
        if chunk_chars + len(sent) > MAX_CHARS and chunk:
            records.append({
                "source_file": str(p.relative_to(EXTRACT_DIR)),
                "chunk_id": chunk_id,
                "text": " ".join(chunk).strip()
            })
            chunk = [sent]
            chunk_chars = len(sent)
            chunk_id += 1
        else:
            chunk.append(sent)
            chunk_chars += len(sent)
    if chunk:
        records.append({
            "source_file": str(p.relative_to(EXTRACT_DIR)),
            "chunk_id": chunk_id,
            "text": " ".join(chunk).strip()
        })

df = pd.DataFrame(records)
df = df[df["text"].str.len() > 40].reset_index(drop=True)

# TF-IDF (jei norėtum naudoti lokaliai)
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=2)
X = vectorizer.fit_transform(df["text"].tolist())

# 5) Sukuriam išvestį
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

with open(Path(OUT_DIR) / "chunks.jsonl", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        rec = {
            "source_file": row["source_file"],
            "chunk_id": int(row["chunk_id"]),
            "text": row["text"]
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

summary = df.groupby("source_file").size().reset_index(name="chunk_count")
summary.to_csv(Path(OUT_DIR) / "sources_summary.csv", index=False, encoding="utf-8")

# Instrukcijos
instructions = """[Čia buvo visas „model_instructions.md“ tekstas, kurį aš tau rašiau anksčiau]"""
(Path(OUT_DIR) / "model_instructions.md").write_text(instructions, encoding="utf-8")

readme = """[Čia buvo README.md tekstas, kurį aš tau rašiau anksčiau]"""
(Path(OUT_DIR) / "README.md").write_text(readme, encoding="utf-8")

# 6) Supakuojam į ZIP
with ZipFile(ZIP_OUT, "w", compression=ZIP_DEFLATED) as z:
    z.write(Path(OUT_DIR) / "chunks.jsonl", arcname="chunks.jsonl")
    z.write(Path(OUT_DIR) / "sources_summary.csv", arcname="sources_summary.csv")
    z.write(Path(OUT_DIR) / "model_instructions.md", arcname="model_instructions.md")
    z.write(Path(OUT_DIR) / "README.md", arcname="README.md")
    for root, dirs, files in os.walk(EXTRACT_DIR):
        for fname in files:
            src = Path(root) / fname
            rel = Path("docs") / "full" / Path(src).relative_to(EXTRACT_DIR)
            z.write(src, arcname=str(rel
