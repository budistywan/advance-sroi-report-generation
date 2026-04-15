# SROI Document Refinement Pipeline v1

Pipeline pengolahan dokumen SROI dari input JSON semantik menjadi dokumen yang harmonized dan siap publikasi.

## Prasyarat

- Python 3.11+
- `pip install pytest openai`
- GitHub Secrets: `OPENAI_API_KEY`

## Struktur

```
data/input/     — file input (pipeline_control_block.json, chapter files)
data/output/    — hasil pipeline (di-commit otomatis oleh Actions)
pipeline/       — Python modules per sprint
tests/          — test suite
ui/             — index.html viewer
```

## Urutan Sprint

```
Sprint 1  — A1 Block ID Injector          (deterministik)
Sprint 2  — A2 Substance Extractor        (LLM)
Sprint 3  — A3 Repair Backlog Builder     (deterministik)
Sprint 4  — B1 Point Builder              (LLM, per bab)
Sprint 5  — B2 Narrative Builder          (LLM, per bab)
Sprint 6  — B3 Narrative Refinement       (LLM, per bab)
Sprint 7  — B4 Visual Asset Builder       (LLM, per bab)
Sprint 8  — D1/D2/D3 Editorial            (LLM, per bab)
Sprint 9  — B5 Consistency Lock           (per bab)
Sprint 10 — C1 Cross-chapter Lock         (LLM)
Sprint 11 — C2 Document Harmonizer        (deterministik)
Sprint 12 — Dry Run + Freeze v1
```

## Menjalankan Pipeline

### Sprint 1 — A1 (otomatis saat push ke data/input/)
```bash
# Manual trigger
gh workflow run run_deterministic.yml -f stage=A1
```

### Sprint 2 — A2 (manual dispatch)
```bash
gh workflow run run_llm.yml -f stage=A2
```

### Sprint 4+ — per bab (manual dispatch)
```bash
gh workflow run run_llm.yml -f stage=B1 -f chapter_id=bab_1
```

### Dry run (tanpa LLM call)
```bash
gh workflow run run_llm.yml -f stage=A2 -f dry_run=true
```

## Input yang Dibutuhkan

Sebelum menjalankan Sprint 1, siapkan di `data/input/`:
- `pipeline_control_block.json`
- `chapter_semantic_bab7.json` dan/atau `chapters_semantic_rest.json`

## Menjalankan Tests Lokal

```bash
pip install pytest openai
pytest tests/ -v
```

## UI

Buka `ui/index.html` di browser. Muat file JSON dari `data/output/` untuk melihat dan mengedit hasil pipeline.
