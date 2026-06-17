import os
import json
import numpy as np

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHUNKS_PATH  = os.path.join(BASE_DIR, "app", "core", "chunks.json")
INDEX_PATH   = os.path.join(BASE_DIR, "app", "core", "faiss_index.bin")
META_PATH    = os.path.join(BASE_DIR, "app", "core", "index_meta.json")

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def main():
    print("\n" + "="*55)
    print("  TinyLogic RAG — Construire Index FAISS")
    print("="*55 + "\n")

    try:
        import faiss
        print("  ✅ FAISS disponibil")
    except ImportError:
        print("  ❌ FAISS lipsește: pip install faiss-cpu")
        return

    try:
        from sentence_transformers import SentenceTransformer
        print("  ✅ SentenceTransformers disponibil")
    except ImportError:
        print("  ❌ sentence-transformers lipsește: pip install sentence-transformers")
        return

    if not os.path.exists(CHUNKS_PATH):
        print(f"  ❌ chunks.json lipsește!")
        print("  Rulează mai întâi: python app/core/parse_courses.py")
        return

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"\n📦 Chunks încărcate: {len(chunks)}")

    print(f"\n🤖 Încărcăm modelul de embeddings...")
    print(f"   Model: {EMBEDDING_MODEL}")
    print(f"   (prima rulare descarcă ~500MB)")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  ✅ Model încărcat")

    print(f"\n🔢 Generăm embeddings pentru {len(chunks)} chunks...")
    texts      = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    print(f"  ✅ Shape: {embeddings.shape}")

    print(f"\n🗂️  Construim indexul FAISS...")
    dimension = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))
    print(f"  ✅ Index: {index.ntotal} vectori")

    faiss.write_index(index, INDEX_PATH)
    print(f"  💾 Index salvat: {INDEX_PATH}")

    meta = {
        "embedding_model": EMBEDDING_MODEL,
        "num_chunks":      len(chunks),
        "dimension":       dimension,
        "index_type":      "IndexFlatIP",
        "chunks_path":     CHUNKS_PATH,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  💾 Metadata salvată: {META_PATH}")

    print(f"\n🧪 Test rapid...")
    test_query = "Cum fac un loop în blocuri?"
    test_emb   = model.encode([test_query], convert_to_numpy=True)
    faiss.normalize_L2(test_emb)
    distances, indices = index.search(test_emb.astype(np.float32), k=3)

    print(f"  Query: '{test_query}'")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        chunk = chunks[idx]
        print(f"  [{i+1}] Score: {dist:.4f} | "
              f"{chunk['metadata'].get('lesson_name','?')[:50]}")

    print("\n" + "="*55)
    print(f"  ✅ INDEX CONSTRUIT!")
    print(f"  Pasul următor: python app/core/rag.py")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()