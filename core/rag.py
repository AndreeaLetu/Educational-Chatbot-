

import os
import json
import time
import numpy as np

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHUNKS_PATH = os.path.join(BASE_DIR, "app", "core", "chunks.json")
INDEX_PATH  = os.path.join(BASE_DIR, "app", "core", "faiss_index.bin")
META_PATH   = os.path.join(BASE_DIR, "app", "core", "index_meta.json")

TOP_K               = 2
RELEVANCE_THRESHOLD = 0.50
LABELS_NO_SEARCH    = {"motivation_or_encouragement"}

GEMINI_MODELS_FALLBACK = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]

DIFFICULTY_STYLE = {
    "beginner": """
Ești un prieten care explică programare unui copil de 8-10 ani.
Vorbești simplu, corect gramatical și cu căldură în limba română.
Folosești TU când te adresezi copilului — niciodată dumneavoastră.
Folosești cuvinte din viața de zi cu zi și analogii simple (ex: variabila = o cutie cu un nume).
Propoziții scurte, maxim 10 cuvinte fiecare.
Răspuns scurt — maxim 4 propoziții — copilul se plictisește repede.
Adaugi un emoji la final 😊
Nu folosești termeni tehnici fără să îi explici imediat cu o analogie.
""",
    "intermediate": """
Vorbești cu un copil care cunoaște bazele programării.
Folosești TU când te adresezi — niciodată dumneavoastră.
Poți folosi termeni tehnici de bază dar îi explici scurt când apar.
Dai un exemplu concret și scurt.
Răspuns concis — maxim 5 propoziții.
Ton prietenos și corect gramatical în limba română.
""",
    "advanced": """
Vorbești cu un copil avansat care înțelege concepte complexe de programare.
Folosești TU când te adresezi — niciodată dumneavoastră.
Folosești terminologie tehnică direct, fără explicații suplimentare.
Răspuns precis și la obiect — maxim 5 propoziții.
Ton corect gramatical în limba română, fără introduceri inutile.
""",
}

LABEL_STYLE = {
    "concept_explanation": (
        "Explică conceptul clar, pe înțelesul copilului. "
        "Folosește o analogie din viața reală dacă ajută. "
        "Nu explica mai mult decât a întrebat."
    ),
    "hint_request": (
        "Oferă UN singur indiciu mic care să îl ajute să gândească singur. "
        "NU da soluția completă. "
        "Închei cu o întrebare scurtă care să îl ghideze mai departe."
    ),
    "solution_request": (
        "Explică soluția pas cu pas, simplu și clar. "
        "Dacă dai cod, scrie-l simplu fără formatare specială. "
        "Explică pe scurt de ce funcționează."
    ),
    "exercise_request": (
        "Propune un exercițiu practic scurt, potrivit nivelului copilului. "
        "Descrie ce trebuie să facă în maxim 2 propoziții. "
        "Nu da și soluția."
    ),
    "code_help": (
        "Identifică problema din codul copilului. "
        "Explică ce a greșit și cum se corectează, simplu și clar. "
        "Dacă dai cod corectat, scrie-l simplu fără formatare specială."
    ),
    "motivation_or_encouragement": (
        "Încurajează copilul cu căldură și entuziasm. "
        "Fii pozitiv și motivant. "
        "Reamintește-i că greșelile fac parte din învățare."
    ),
}

COMMON_RULES = """
REGULI STRICTE — respectă-le întotdeauna:
- Vorbești cu copilul la TU (persoana a II-a singular) — niciodată dumneavoastră
- NU saluta la început — intră direct în răspuns
- FĂRĂ markdown, FĂRĂ asteriscuri (*), FĂRĂ backticks (```)
- Dacă dai un exemplu de cod, scrie-l simplu pe o linie nouă, fără formatare
- Scrie doar text simplu și curat în română
- NU repeta întrebarea copilului
- Dacă există context de conversație anterioară, ține cont de el
"""

class RAGEngine:

    def __init__(self, gemini_api_key=None):
        self.index          = None
        self.chunks         = None
        self.embedder       = None
        self.gemini         = None
        self.meta           = None
        self._gemini_client = None

        self._load_index()
        self._load_embedder()
        self._load_gemini()

    def _load_index(self):
        try:
            import faiss
            if not os.path.exists(INDEX_PATH):
                raise FileNotFoundError(f"Index lipsește: {INDEX_PATH}")
            self.index = faiss.read_index(INDEX_PATH)
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            with open(META_PATH, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
            print(f"   Index FAISS: {self.index.ntotal} vectori")
        except ImportError:
            print("  FAISS lipsește: pip install faiss-cpu")
            raise

    def _load_embedder(self):
        try:
            from sentence_transformers import SentenceTransformer
            model_name    = self.meta.get(
                "embedding_model",
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            )
            self.embedder = SentenceTransformer(model_name)
            print(f" Embedder încărcat")
        except ImportError:
            print("sentence-transformers lipsește")
            raise

    def _load_gemini(self):
        try:
            from google import genai

            key = ""
            env_path = os.path.join(BASE_DIR, ".env")
            if os.path.exists(env_path):
                with open(env_path) as f:
                    for line in f:
                        if line.startswith("GEMINI_API_KEY="):
                            key = line.strip().split("=", 1)[1]
                            break

            if not key:
                key = os.environ.get("GEMINI_API_KEY", "")

            if not key:
                print("  GEMINI_API_KEY lipsește în .env")
                self.gemini = None
                return

            self._gemini_client = genai.Client(api_key=key)
            self.gemini = "gemini"
            print(f" Gemini inițializat ({GEMINI_MODELS_FALLBACK[0]})")

        except ImportError:
            print("  google-genai lipsește: pip install google-genai")
            self.gemini = None
        except Exception as e:
            print(f"   Eroare Gemini: {e}")
            self.gemini = None

    def search(self, question, k=TOP_K, filter_course=None):
        import faiss

        query_emb = self.embedder.encode(
            [question], convert_to_numpy=True
        ).astype(np.float32)
        faiss.normalize_L2(query_emb)

        k_search = k * 3 if filter_course else k
        distances, indices = self.index.search(query_emb, k=k_search)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            if filter_course:
                if chunk["metadata"].get("language", "").lower() != filter_course.lower():
                    continue
            results.append({
                "chunk":  chunk,
                "score":  float(dist),
                "text":   chunk["text"],
                "lesson": chunk["metadata"].get("lesson_name", ""),
                "course": chunk["metadata"].get("course", ""),
            })
            if len(results) >= k:
                break

        return results

    def _format_history(self, history):
        if not history:
            return ""

        lines = ["\nCONTEXTUL CONVERSAȚIEI ANTERIOARE:"]
        for msg in history[-6:]:
            role    = "Copil" if msg.get("role") == "user" else "Boty"
            content = msg.get("content", "").strip()
            if content:
                lines.append(f"{role}: {content}")
        lines.append("")
        return "\n".join(lines)

    def _build_prompt(self, question, label, difficulty,
                      retrieved=None, history=None):
        diff_style    = DIFFICULTY_STYLE.get(difficulty, DIFFICULTY_STYLE["intermediate"])
        label_style   = LABEL_STYLE.get(label, LABEL_STYLE["concept_explanation"])
        history_text  = self._format_history(history)

        if retrieved:
            context_section = (
                "\nINFORMAȚII DIN CURS (bazează-te pe acestea dar reformulează):\n" +
                "\n\n".join([
                    f"[{r['course']}, {r['lesson']}]\n{r['text']}"
                    for r in retrieved
                ]) + "\n"
            )
        else:
            context_section = ""

        return f"""Ești TinyLogic Assistant, asistent educațional pentru copii care învață programare.
TinyLogic predă programare prin blocuri vizuale, Python și C.

{diff_style}

SARCINA TA: {label_style}
{history_text}{context_section}
ÎNTREBAREA CURENTĂ: {question}

{COMMON_RULES}

Răspunde în EXACT acest format — respectă etichetele:
GAND: (scrie 2 propoziții scurte în română la persoana I — ce gândești înainte să răspunzi, ținând cont de contextul conversației dacă există)
RASPUNS: (scrie răspunsul final — maxim 4-5 propoziții simple, fără markdown)"""

    def _build_prompt_motivation(self, question, difficulty, history=None):
        diff_style   = DIFFICULTY_STYLE.get(difficulty, DIFFICULTY_STYLE["beginner"])
        history_text = self._format_history(history)

        return f"""Ești TinyLogic Assistant, asistent educațional pentru copii care învață programare.

{diff_style}
{history_text}
Un copil are nevoie de încurajare. Mesajul său: "{question}"

{COMMON_RULES}
- Fii cald, pozitiv și motivant
- Maxim 3 propoziții
- NU menționezi cursuri sau surse

Răspunde în EXACT acest format:
GAND: (ce gândești în 1-2 propoziții scurte în română)
RASPUNS: (răspunsul motivațional — maxim 3 propoziții)"""

    def _call_gemini(self, prompt):

        if self.gemini != "gemini":
            return "", None

        from google.genai import types

        for model in GEMINI_MODELS_FALLBACK:
            try:
                response = self._gemini_client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.5,
                        max_output_tokens=400,
                    )
                )
                raw = response.text.strip()

                raw = raw.replace("```python", "").replace("```", "")
                raw = raw.replace("**", "").replace("__", "")

                thinking    = ""
                answer_text = ""

                if "RASPUNS:" in raw:
                    parts       = raw.split("RASPUNS:", 1)
                    gand_part   = parts[0]
                    thinking    = gand_part.split("GAND:", 1)[1].strip() \
                                  if "GAND:" in gand_part else gand_part.strip()
                    answer_text = parts[1].strip()
                else:
                    answer_text = raw

                return thinking, answer_text

            except Exception as e:
                err = str(e)
                if "429" in err or "503" in err:
                    print(f"   {model} indisponibil — încerc următorul")
                    time.sleep(2)
                    continue
                else:
                    print(f"  Eroare Gemini ({model}): {e}")
                    return "", None

        print("  ⚠️  Toate modelele Gemini indisponibile")
        return "", None

    def answer(self, question, label="concept_explanation",
               difficulty="intermediate", course=None, history=None):

        if label in LABELS_NO_SEARCH:
            if self.gemini == "gemini":
                prompt              = self._build_prompt_motivation(
                    question, difficulty, history
                )
                thinking, answer_text = self._call_gemini(prompt)
                if not answer_text:
                    answer_text = "Continuă să înveți — fiecare greșeală te face mai bun! 💪"
                    thinking    = ""
            else:
                answer_text = "Continuă să înveți — fiecare greșeală te face mai bun! 💪"
                thinking    = ""

            return {
                "answer":      answer_text,
                "sources":     [],
                "label":       label,
                "difficulty":  difficulty,
                "chunks_used": 0,
                "best_score":  0,
                "used_course": False,
                "mode":        "motivation",
                "thinking":    [thinking] if thinking else [],
            }

        retrieved   = self.search(question, k=TOP_K, filter_course=course)
        best_score  = max([r["score"] for r in retrieved], default=0)
        use_context = best_score >= RELEVANCE_THRESHOLD

        thinking    = ""
        answer_text = ""
        mode        = ""

        if self.gemini == "gemini":
            if use_context and retrieved:
                prompt = self._build_prompt(
                    question, label, difficulty, retrieved, history
                )
                mode = "from_course"
            else:
                prompt = self._build_prompt(
                    question, label, difficulty, None, history
                )
                mode = "from_knowledge"

            thinking, answer_text = self._call_gemini(prompt)

            if not answer_text:
                if retrieved:
                    raw         = retrieved[0]["text"]
                    sentences   = raw.split(".")
                    answer_text = ". ".join(sentences[:2]).strip() + "."
                    mode        = "fallback_text"
                    thinking    = ""
                else:
                    answer_text = "Nu am găsit informații despre asta în cursurile TinyLogic."
                    mode        = "no_result"
                    thinking    = ""
        else:
            if retrieved:
                raw         = retrieved[0]["text"]
                sentences   = raw.split(".")
                answer_text = ". ".join(sentences[:2]).strip() + "."
                mode        = "fallback_text"
            else:
                answer_text = "Gemini indisponibil și nu am găsit informații în curs."
                mode        = "no_result"

        sources = []
        if use_context and mode == "from_course":
            sources = [
                {
                    "course":  r["course"],
                    "lesson":  r["lesson"],
                    "score":   round(r["score"], 4),
                    "excerpt": r["text"][:100] + "..."
                }
                for r in retrieved
            ]

        return {
            "answer":      answer_text,
            "sources":     sources,
            "label":       label,
            "difficulty":  difficulty,
            "chunks_used": len(retrieved),
            "best_score":  round(best_score, 4),
            "used_course": use_context,
            "mode":        mode,
            "thinking":    [thinking] if thinking else [],
        }
def test_rag():
    print("\n" + "="*55)
    print("  TinyLogic RAG — Test cu Istoric")
    print("="*55 + "\n")

    engine  = RAGEngine()
    history = []

    tests = [
        {"question": "Ce este un loop în Python?",
         "label": "concept_explanation", "difficulty": "beginner"},
        {"question": "Dar cum îl opresc?",
         "label": "concept_explanation", "difficulty": "beginner"},
        {"question": "Nu înțeleg, e prea greu",
         "label": "motivation_or_encouragement", "difficulty": "beginner"},
    ]

    for i, test in enumerate(tests):
        print(f"\n{'─'*55}")
        print(f"Test {i+1}: [{test['label']}] [{test['difficulty']}]")
        print(f"  Q: {test['question']}")
        print(f"{'─'*55}")

        result = engine.answer(**test, history=history)

        history.append({"role": "user",      "content": test["question"]})
        history.append({"role": "assistant", "content": result["answer"]})
        if len(history) > 6:
            history = history[-6:]

        if result.get("thinking"):
            print(f"  💭 {result['thinking'][0]}")
        print(f"  A: {result['answer']}")
        print(f"  Mod: {result['mode']}")

    print(f"\n{'='*55}\n")


if __name__ == "__main__":
    test_rag()