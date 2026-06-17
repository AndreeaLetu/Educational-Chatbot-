import os
import json
import re

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COURSES_DIR = os.path.join(BASE_DIR, "app", "data", "courses")
OUTPUT_PATH = os.path.join(BASE_DIR, "app", "core", "chunks.json")

COURSE_TYPES = {
    "blocuri": {"name": "Blocuri", "language": "blocuri"},
    "python":  {"name": "Python",  "language": "python"},
    "c":       {"name": "C",       "language": "c"},
}


def clean_text(text):
    if not text:
        return ""

    def replace_block(match):
        parts = match.group(1).split("|")
        if len(parts) == 1:
            return f"blocul {parts[0].upper()}"
        elif len(parts) == 2:
            return f"blocul {parts[1].upper()}"
        elif len(parts) == 3:
            return f"blocul {parts[1].upper()} cu valoarea {parts[2]}"
        return match.group(0)

    text = re.sub(r"\{\{([^}]+)\}\}", replace_block, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_lesson(lesson, course_type):
    chunks      = []
    lesson_name = lesson.get("LessonName", "Lecție necunoscută")
    order_idx   = lesson.get("OrderIndex", 0)
    content     = lesson.get("ContentJson", {})
    title       = content.get("Title", "")
    sections    = content.get("Sections", [])

    base_meta = {
        "course":      course_type["name"],
        "language":    course_type["language"],
        "lesson_name": lesson_name,
        "order_index": order_idx,
        "title":       title,
    }

    full_parts = [f"Lecția: {lesson_name}"]
    if title:
        full_parts.append(f"Titlu: {title}")
    for section in sections:
        heading = section.get("Heading", "")
        text    = clean_text(section.get("Text", ""))
        if heading:
            full_parts.append(f"{heading}: {text}")
        else:
            full_parts.append(text)

    full_text = " | ".join(full_parts)
    if full_text.strip():
        chunks.append({
            "text":       full_text,
            "chunk_type": "lesson_full",
            "metadata":   {**base_meta, "chunk_type": "lesson_full"}
        })

    for i, section in enumerate(sections):
        heading = section.get("Heading", "")
        text    = clean_text(section.get("Text", ""))
        if not text:
            continue
        section_text = f"{lesson_name}"
        if heading:
            section_text += f" — {heading}"
        section_text += f": {text}"
        chunks.append({
            "text":       section_text,
            "chunk_type": "section",
            "metadata":   {
                **base_meta,
                "chunk_type":    "section",
                "section_index": i,
                "heading":       heading,
            }
        })

    exercises = []
    if "Exercise" in content:
        exercises = [content["Exercise"]]
    elif "Exercises" in content:
        exercises = content["Exercises"]

    for i, exercise in enumerate(exercises):
        desc     = clean_text(exercise.get("Description", ""))
        hint     = clean_text(exercise.get("Hint", ""))
        level    = exercise.get("Level", "")
        ex_title = exercise.get("Title", f"Exercițiul {i+1}")

        if not desc:
            continue

        ex_text = f"{lesson_name} — {ex_title}"
        if level:
            ex_text += f" (nivel {level})"
        ex_text += f": {desc}"
        if hint:
            ex_text += f" Indiciu: {hint}"

        chunks.append({
            "text":       ex_text,
            "chunk_type": "exercise",
            "metadata":   {
                **base_meta,
                "chunk_type":     "exercise",
                "exercise_index": i,
                "exercise_title": ex_title,
                "exercise_level": level,
                "has_hint":       bool(hint),
            }
        })

    return chunks


def parse_course_file(filepath, course_type):
    with open(filepath, "r", encoding="utf-8-sig") as f:
        lessons = json.load(f)
    all_chunks = []
    for lesson in lessons:
        all_chunks.extend(parse_lesson(lesson, course_type))
    return all_chunks


def main():
    print("\n" + "="*55)
    print("  TinyLogic RAG — Parsare Cursuri")
    print("="*55 + "\n")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    os.makedirs(COURSES_DIR, exist_ok=True)

    json_files = [f for f in os.listdir(COURSES_DIR) if f.endswith(".json")] \
        if os.path.exists(COURSES_DIR) else []

    if not json_files:
        print(f"  ⚠️  Nu am găsit fișiere JSON în: {COURSES_DIR}")
        print(f"  Pune fișierele JSON ale cursurilor acolo și rulează din nou.")
        print(f"\n  Structura așteptată:")
        print(f"  app/data/courses/blocuri.json")
        print(f"  app/data/courses/python.json")
        print(f"  app/data/courses/c.json")
        return []

    all_chunks = []

    for json_file in json_files:
        course_key  = json_file.replace(".json", "").lower()
        course_type = COURSE_TYPES.get(
            course_key,
            {"name": course_key.capitalize(), "language": course_key}
        )
        filepath = os.path.join(COURSES_DIR, json_file)
        print(f"📖 Parsăm: {json_file} ({course_type['name']})")

        try:
            chunks = parse_course_file(filepath, course_type)
            all_chunks.extend(chunks)

            n_full      = sum(1 for c in chunks if c["chunk_type"] == "lesson_full")
            n_sections  = sum(1 for c in chunks if c["chunk_type"] == "section")
            n_exercises = sum(1 for c in chunks if c["chunk_type"] == "exercise")

            print(f"   Lecții: {n_full} | Secțiuni: {n_sections} | Exerciții: {n_exercises}")
            print(f"   Total chunks: {len(chunks)}\n")
        except Exception as e:
            print(f"   ❌ Eroare: {e}\n")

    for i, chunk in enumerate(all_chunks):
        chunk["id"] = i

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print("="*55)
    print(f"  ✅ PARSARE COMPLETĂ!")
    print(f"  Total chunks: {len(all_chunks)}")
    print(f"  Salvat în: {OUTPUT_PATH}")
    print(f"\n  Pasul următor: python app/core/build_index.py")
    print("="*55 + "\n")

    return all_chunks


if __name__ == "__main__":
    main()