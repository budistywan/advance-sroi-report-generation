"""
A1 — Block ID Injector
Sprint 1 | SROI Document Refinement Pipeline v1
Status: FROZEN spec

Menambahkan tiga field ke setiap block:
  - block_ref        : {chapter_id}.B{zero_padded_index}  (chapter-local)
  - block_fingerprint: {chapter_id}__{type}__{normalized_prefix}  (document-wide)
  - original_index   : int, 0-based, posisi asli di array input

Rules enforced:
  R-A1-01  Identity append-only (downstream constraint — documented)
  R-A1-02  Fingerprint uniqueness document-wide (collision = fatal)
  R-A1-03  block_ref unik chapter-wide
  R-A1-04  Fingerprint content-derived, bukan index
  R-A1-05  Injection non-destructive
  R-A1-06  Empty-content blocks → warning, fingerprint ...__empty
  R-A1-07  Collision fatal, proses stop
  R-A1-08  No auto-disambiguation (no _2, _3 suffix)
  R-A1-09  Control block valid sebelum proses
  R-A1-10  Program-agnostic
  R-A1-11  Non-content blocks → structural fingerprint
  R-A1-12  Non-content blocks excluded dari collision check content
"""

from __future__ import annotations

import json
import re
import sys
import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

ACTIVITY = "block_id_injector"
PIPELINE_VERSION = "v1"

DEFAULT_NON_CONTENT_BLOCK_TYPES = ["divider", "divider_thick", "spacer"]

VALID_SUBSTANCE_MODES   = {"confirmed", "discovery"}
VALID_SUBSTANCE_BASES   = {"manual_audit", "backlog_derived", "config_declared", "unknown"}

CONTENT_SOURCE_PRIORITY = [
    "text",
    ("items", 0, "text"),
    ("items", 0, "label"),   # fix: metric_card dan block dengan items.label
    "title",                  # fix: pindah sebelum headers[0] agar judul block diutamakan
    ("headers", 0),
    "caption",
    ("rows", 0, 0),
    "label",
    "value",
]

# Regex patterns (from Deliverable 4 Validation Spec)
RE_BLOCK_REF_CONTENT    = re.compile(r"^[A-Za-z0-9_-]+\.B[0-9]{3,}$")
RE_BLOCK_FP_CONTENT     = re.compile(r"^[A-Za-z0-9_-]+__[A-Za-z0-9_-]+__[a-z0-9_]+$")
RE_BLOCK_FP_STRUCTURAL  = re.compile(r"^[A-Za-z0-9_-]+__[A-Za-z0-9_-]+__structural_[0-9]+$")

# Punctuation to strip during normalization (step 5)
RE_PUNCTUATION = re.compile(r"[.,!?:;\"'()\[\]{}/\\=\-_]")


# ─────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────

def extract_content_source(block: dict) -> tuple[str | None, bool, str | None]:
    """
    Cari primary content source dari block sesuai priority order.
    Returns (raw_text, found, source_key).
    source_key adalah label human-readable dari source yang dipakai.
    """
    for source in CONTENT_SOURCE_PRIORITY:
        try:
            if isinstance(source, str):
                val = block.get(source)
                if val is not None and str(val).strip():
                    return str(val), True, source
            elif isinstance(source, tuple):
                # nested access: e.g. ("items", 0, "text")
                val = block
                for key in source:
                    if isinstance(key, int):
                        if not isinstance(val, list) or len(val) <= key:
                            val = None
                            break
                        val = val[key]
                    else:
                        if not isinstance(val, dict):
                            val = None
                            break
                        val = val.get(key)
                if val is not None and str(val).strip():
                    # Bentuk label: "items[0].text", "headers[0]", "rows[0][0]"
                    label = _source_tuple_label(source)
                    return str(val), True, label
        except (KeyError, IndexError, TypeError):
            continue
    return None, False, None


def _source_tuple_label(source: tuple) -> str:
    """Bentuk label readable dari tuple source path."""
    parts = []
    for i, key in enumerate(source):
        if isinstance(key, int):
            parts.append(f"[{key}]")
        else:
            # tambahkan titik jika sebelumnya ada index atau field lain
            if parts:
                parts.append(f".{key}")
            else:
                parts.append(key)
    return "".join(parts)


def normalize_prefix(raw: str) -> str:
    """
    Normalization algorithm (Deliverable 1, 9 steps):
    1. lowercase
    2. trim whitespace
    3. newline/tab → spasi
    4. collapse multiple spaces → satu spasi
    5. hapus tanda baca  (underscore dikonversi ke spasi dulu agar
       survive menjadi underscore kembali di step 6 — edge case fix)
    6. spasi → underscore
    7. hapus karakter non-alphanumeric selain underscore
    8. ambil 40 karakter pertama
    9. jika kosong → 'empty'
    """
    s = raw.lower()
    s = s.strip()
    s = re.sub(r"[\n\t\r]", " ", s)
    # Step 5 pre-pass: underscore → spasi, agar tidak ikut terhapus
    # oleh RE_PUNCTUATION, lalu dibangun ulang sebagai underscore di step 6
    s = s.replace("_", " ")
    s = re.sub(r" +", " ", s)
    s = RE_PUNCTUATION.sub("", s)
    s = re.sub(r" +", " ", s)   # collapse ulang setelah strip punctuation
    s = s.replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = s[:40]
    return s if s else "empty"


def build_block_ref(chapter_id: str, one_based_index: int) -> str:
    """Format: {chapter_id}.B{zero_padded, min 3 digit}"""
    return f"{chapter_id}.B{one_based_index:03d}"


def build_content_fingerprint(chapter_id: str, block_type: str, normalized_prefix: str) -> str:
    return f"{chapter_id}__{block_type}__{normalized_prefix}"


def build_structural_fingerprint(chapter_id: str, block_type: str, original_index: int) -> str:
    return f"{chapter_id}__{block_type}__structural_{original_index}"


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────

def validate_control_block(pcb: dict) -> list[str]:
    """
    Validate pipeline_control_block.
    Returns list of error messages. Empty = valid.
    """
    errors = []
    if not isinstance(pcb, dict):
        return ["pipeline_control_block harus berupa object"]

    for field in ["document_id", "program_code", "substance_mode", "substance_basis"]:
        val = pcb.get(field)
        if not val or not str(val).strip():
            errors.append(f"pipeline_control_block.{field} wajib diisi dan tidak boleh kosong")

    mode = pcb.get("substance_mode")
    if mode and mode not in VALID_SUBSTANCE_MODES:
        errors.append(
            f"substance_mode '{mode}' tidak valid. Harus: {', '.join(sorted(VALID_SUBSTANCE_MODES))}"
        )

    basis = pcb.get("substance_basis")
    if basis and basis not in VALID_SUBSTANCE_BASES:
        errors.append(
            f"substance_basis '{basis}' tidak valid. Harus: {', '.join(sorted(VALID_SUBSTANCE_BASES))}"
        )

    # cross-field: confirmed mode constraints
    if mode == "confirmed":
        if not pcb.get("substance_chapter_id"):
            errors.append(
                "substance_mode=confirmed membutuhkan substance_chapter_id yang terisi"
            )
        if basis == "unknown":
            errors.append(
                "substance_mode=confirmed tidak boleh pakai substance_basis=unknown"
            )

    return errors


def validate_chapter(chapter: dict, chapter_idx: int) -> list[str]:
    errors = []
    chapter_id = chapter.get("chapter_id")
    if not chapter_id or not str(chapter_id).strip():
        errors.append(f"chapters[{chapter_idx}]: chapter_id wajib ada dan tidak kosong")
    blocks = chapter.get("blocks")
    if not isinstance(blocks, list):
        errors.append(
            f"chapters[{chapter_idx}] ({chapter_id or '?'}): 'blocks' harus berupa array"
        )
    return errors


def validate_block(block: Any, chapter_id: str, block_idx: int) -> list[str]:
    errors = []
    if not isinstance(block, dict):
        errors.append(
            f"{chapter_id}[{block_idx}]: block harus berupa object, dapat: {type(block).__name__}"
        )
        return errors
    if not block.get("type"):
        errors.append(f"{chapter_id}[{block_idx}]: block wajib punya field 'type'")
    return errors


# ─────────────────────────────────────────────
# CORE INJECTION
# ─────────────────────────────────────────────

def inject_chapter(
    chapter: dict,
    non_content_types: set[str],
    doc_fingerprints: dict[str, str],   # fingerprint → block_ref, untuk collision check
) -> tuple[dict, list[dict], list[dict]]:
    """
    Proses satu chapter: inject block_ref, block_fingerprint, original_index.

    Returns:
        (enriched_chapter, warnings, failures)

    Jika ada fatal failure (collision), failures akan berisi entry dan
    enriched_chapter dikembalikan dalam kondisi parsial (tidak boleh dipakai).
    """
    chapter_id = chapter["chapter_id"]
    blocks_in = chapter["blocks"]
    warnings: list[dict] = []
    failures: list[dict] = []

    enriched_blocks = []

    for original_index, block in enumerate(blocks_in):
        block_type = block.get("type", "unknown")
        one_based  = original_index + 1
        block_ref  = build_block_ref(chapter_id, one_based)

        # ── Non-content block ─────────────────────────
        if block_type in non_content_types:
            fingerprint = build_structural_fingerprint(chapter_id, block_type, original_index)
            enriched = {
                "block_ref": block_ref,
                "block_fingerprint": fingerprint,
                "original_index": original_index,
                **block,
            }
            enriched_blocks.append(enriched)
            continue

        # ── Content block ─────────────────────────────
        raw_content, found, source_key = extract_content_source(block)

        if not found:
            # R-A1-06: empty content → warning, fingerprint __empty
            normalized = "empty"
            warnings.append({
                "level": "warning",
                "code": "EMPTY_CONTENT_BLOCK",
                "block_ref": block_ref,
                "chapter_id": chapter_id,
                "original_index": original_index,
                "message": (
                    f"{block_ref}: tidak ada content source ditemukan "
                    f"(type={block_type}). Fingerprint akan memakai '__empty'."
                ),
            })
        else:
            normalized = normalize_prefix(raw_content)
            primary_is_text = source_key == "text"
            if not primary_is_text:
                # 'text' tidak tersedia atau kosong — prefix dari source lain
                warnings.append({
                    "level": "warning",
                    "code": "PRIMARY_SOURCE_ABSENT",
                    "block_ref": block_ref,
                    "chapter_id": chapter_id,
                    "original_index": original_index,
                    "source_used": source_key,
                    "message": (
                        f"{block_ref}: field 'text' tidak tersedia, "
                        f"prefix diambil dari '{source_key}'."
                    ),
                })

        fingerprint = build_content_fingerprint(chapter_id, block_type, normalized)

        # ── R-A1-07 & R-A1-08: Collision check ───────
        if fingerprint in doc_fingerprints:
            existing_ref = doc_fingerprints[fingerprint]
            failures.append({
                "level": "fatal",
                "code": "FINGERPRINT_COLLISION",
                "block_ref": block_ref,
                "collides_with": existing_ref,
                "fingerprint": fingerprint,
                "chapter_id": chapter_id,
                "original_index": original_index,
                "message": (
                    f"COLLISION FATAL: {block_ref} menghasilkan fingerprint "
                    f"'{fingerprint}' yang sama dengan {existing_ref}. "
                    f"Tidak ada auto-disambiguation. Proses dihentikan."
                ),
            })
            # Langsung return — stop processing
            enriched_chapter = {**chapter, "blocks": enriched_blocks}
            return enriched_chapter, warnings, failures

        doc_fingerprints[fingerprint] = block_ref

        enriched = {
            "block_ref": block_ref,
            "block_fingerprint": fingerprint,
            "original_index": original_index,
            **block,
        }
        enriched_blocks.append(enriched)

    enriched_chapter = {
        "chapter_id": chapter.get("chapter_id"),
        "chapter_type": chapter.get("chapter_type", None),
        "blocks": enriched_blocks,
    }
    return enriched_chapter, warnings, failures


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────

def run(input_data: dict) -> dict:
    """
    Jalankan A1 Block ID Injector.

    Input dict harus mengandung:
      - pipeline_control_block
      - semantic_chapters

    Returns block_identity_packet dict.
    """
    all_warnings: list[dict] = []
    all_failures: list[dict] = []

    # ── Step 1: Validate control block ────────
    pcb = input_data.get("pipeline_control_block", {})
    pcb_errors = validate_control_block(pcb)
    if pcb_errors:
        for err in pcb_errors:
            all_failures.append({
                "level": "fatal",
                "code": "CONTROL_BLOCK_INVALID",
                "message": err,
            })
        return _emit_failure_packet(input_data, all_warnings, all_failures)

    non_content_types = set(
        pcb.get("non_content_block_types", DEFAULT_NON_CONTENT_BLOCK_TYPES)
    )

    # ── Step 2: Validate semantic_chapters ────
    semantic_chapters = input_data.get("semantic_chapters", [])
    if not isinstance(semantic_chapters, list) or len(semantic_chapters) == 0:
        all_failures.append({
            "level": "fatal",
            "code": "EMPTY_CHAPTERS",
            "message": "semantic_chapters tidak ditemukan atau kosong.",
        })
        return _emit_failure_packet(input_data, all_warnings, all_failures)

    for ci, chapter in enumerate(semantic_chapters):
        ch_errors = validate_chapter(chapter, ci)
        if ch_errors:
            for err in ch_errors:
                all_failures.append({
                    "level": "fatal",
                    "code": "CHAPTER_INVALID",
                    "message": err,
                })
            return _emit_failure_packet(input_data, all_warnings, all_failures)

        blocks = chapter.get("blocks", [])
        for bi, block in enumerate(blocks):
            b_errors = validate_block(block, chapter.get("chapter_id", f"ch_{ci}"), bi)
            if b_errors:
                for err in b_errors:
                    all_failures.append({
                        "level": "fatal",
                        "code": "BLOCK_INVALID",
                        "message": err,
                    })
                return _emit_failure_packet(input_data, all_warnings, all_failures)

        if chapter.get("chapter_type") is None:
            all_warnings.append({
                "level": "warning",
                "code": "CHAPTER_TYPE_NULL",
                "chapter_id": chapter.get("chapter_id"),
                "message": (
                    f"chapter '{chapter.get('chapter_id')}': chapter_type adalah null."
                ),
            })

    # ── Step 3–5: Inject per chapter ──────────
    doc_fingerprints: dict[str, str] = {}   # accumulates across chapters
    enriched_chapters = []
    total_blocks = 0
    empty_content_blocks = 0

    for chapter in semantic_chapters:
        # deep copy agar tidak mutasi input
        chapter_copy = copy.deepcopy(chapter)
        enriched_ch, ch_warnings, ch_failures = inject_chapter(
            chapter_copy, non_content_types, doc_fingerprints
        )
        all_warnings.extend(ch_warnings)
        all_failures.extend(ch_failures)

        empty_content_blocks += sum(
            1 for w in ch_warnings if w.get("code") == "EMPTY_CONTENT_BLOCK"
        )

        if ch_failures:
            # Fatal in this chapter — stop everything
            return _emit_failure_packet(
                input_data, all_warnings, all_failures,
                partial_chapters=enriched_chapters + [enriched_ch],
            )

        enriched_chapters.append(enriched_ch)
        total_blocks += len(enriched_ch.get("blocks", []))

    # ── Step 6: Emit final packet ──────────────
    collision_count = sum(
        1 for f in all_failures if f.get("code") == "FINGERPRINT_COLLISION"
    )

    packet = {
        "activity": ACTIVITY,
        "document_id": pcb["document_id"],
        "program_code": pcb["program_code"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "chapters": enriched_chapters,
        "summary": {
            "total_chapters": len(enriched_chapters),
            "total_blocks": total_blocks,
            "fingerprint_collisions": collision_count,
            "empty_content_blocks": empty_content_blocks,
            "warnings_count": len(all_warnings),
            "failure_count": len(all_failures),
        },
        "warnings": all_warnings,
        "failures": all_failures,
    }

    return packet


def _emit_failure_packet(
    input_data: dict,
    warnings: list[dict],
    failures: list[dict],
    partial_chapters: list | None = None,
) -> dict:
    """Emit packet saat fatal failure terjadi."""
    pcb = input_data.get("pipeline_control_block", {})
    return {
        "activity": ACTIVITY,
        "document_id": pcb.get("document_id", "unknown"),
        "program_code": pcb.get("program_code", "unknown"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "status": "failed",
        "chapters": partial_chapters or [],
        "summary": {
            "total_chapters": len(partial_chapters) if partial_chapters else 0,
            "total_blocks": 0,
            "fingerprint_collisions": sum(
                1 for f in failures if f.get("code") == "FINGERPRINT_COLLISION"
            ),
            "empty_content_blocks": 0,
            "warnings_count": len(warnings),
            "failure_count": len(failures),
        },
        "warnings": warnings,
        "failures": failures,
    }


# ─────────────────────────────────────────────
# CLI ENTRYPOINT
# ─────────────────────────────────────────────

def main():
    """
    Usage:
      python a1_block_id_injector.py <input_file> [output_file]

    input_file  : JSON dengan keys pipeline_control_block + semantic_chapters
    output_file : path output (default: data/output/block_identity_packet.json)
    """
    if len(sys.argv) < 2:
        print(
            "Usage: python a1_block_id_injector.py <input_file> [output_file]",
            file=sys.stderr,
        )
        sys.exit(1)

    input_path  = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else \
                  Path("data/output/block_identity_packet.json")

    if not input_path.exists():
        print(f"ERROR: Input file tidak ditemukan: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        input_data = json.load(f)

    packet = run(input_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(packet, f, ensure_ascii=False, indent=2)

    # Summary ke stdout
    summary = packet.get("summary", {})
    status  = "FAILED" if packet.get("failures") else "OK"
    print(f"[A1] Status        : {status}")
    print(f"[A1] Chapters      : {summary.get('total_chapters', 0)}")
    print(f"[A1] Blocks        : {summary.get('total_blocks', 0)}")
    print(f"[A1] Collisions    : {summary.get('fingerprint_collisions', 0)}")
    print(f"[A1] Warnings      : {summary.get('warnings_count', 0)}")
    print(f"[A1] Failures      : {summary.get('failure_count', 0)}")
    print(f"[A1] Output        : {output_path}")

    if packet.get("failures"):
        print("\nFATAL FAILURES:", file=sys.stderr)
        for f in packet["failures"]:
            print(f"  [{f.get('code')}] {f.get('message')}", file=sys.stderr)
        sys.exit(1)

    if packet.get("warnings"):
        print("\nWarnings:")
        for w in packet["warnings"]:
            print(f"  [{w.get('code')}] {w.get('message')}")


if __name__ == "__main__":
    main()
