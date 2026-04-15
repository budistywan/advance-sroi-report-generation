"""
A2 — Substance Extractor
Sprint 2 | SROI Document Refinement Pipeline v1
Status: FROZEN spec

Dua output:
  - substance_registry.json
  - substance_guardrails.json

Two-pass LLM strategy:
  Pass 1 — Extraction + Labeling
    Input : blocks dari substance chapter
    Output: daftar elemen (element_type, label, summary, provenance,
            evidence_status, use_affordances, guardrail_notes)

  Pass 2 — Scoring + Guardrail enrichment
    Input : hasil Pass 1 + context guardrails global
    Output: materiality_score, reusability_score, priority per elemen
            + global_guardrails dan element_guardrails

Rules enforced:
  R-A2-01  Source chapter dari pipeline_control_block, bukan hardcoded
  R-A2-02  Extraction selektif, bukan exhaustive copy
  R-A2-03  Provenance wajib (source_block_refs / fingerprints / types)
  R-A2-04  evidence_status wajib per elemen
  R-A2-05  use_affordances wajib per elemen
  R-A2-06  guardrail_notes wajib per elemen
  R-A2-07  Non-content blocks tidak boleh jadi standalone elements
  R-A2-08  Mixed-source element diperbolehkan
  R-A2-09  Granularity reusable
  R-A2-10  Evidence status = weakest safe interpretation
  R-A2-11  use_affordances controlled extensible (core + x_...)
  R-A2-12  element_type=other wajib element_type_note
  R-A2-13  Priority dari skor, bukan intuisi
  R-A2-14  Guardrail notes spesifik terhadap risiko
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

ACTIVITY         = "substance_extractor"
PIPELINE_VERSION = "v1"
LLM_MODEL        = "gpt-4o"

VALID_ELEMENT_TYPES = {
    "evaluative_mandate", "program_positioning", "scope_definition",
    "stakeholder_structure", "investment_structure", "output_structure",
    "outcome_structure", "adjustment_logic", "monetization_logic",
    "evaluative_metric", "interpretive_finding", "learning_signal",
    "problem_signal", "ideal_state_signal", "strategy_signal",
    "program_structure", "other",
}

VALID_EVIDENCE_STATUSES = {"final", "proxy", "pending", "inferred", "mixed"}

CORE_AFFORDANCES = {
    "opening_mandate", "program_positioning", "scope_definition",
    "methodological_reference", "problem_justification", "baseline_context",
    "ideal_state_anchor", "strategy_alignment", "implementation_reference",
    "outcome_reference", "adjustment_reference", "monetization_reference",
    "learning_anchor", "closing_summary", "recommendation_basis",
    "visual_candidate",
}

RE_AFFORDANCE_EXT = re.compile(r"^x_[a-z0-9_]+$")
RE_ELEMENT_ID     = re.compile(r"^SUB-\d+$")
RE_GUARDRAIL_ID   = re.compile(r"^SG-\d+$")

NON_CONTENT_BLOCK_TYPES_DEFAULT = {"divider", "divider_thick", "spacer"}


# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────

SYSTEM_PASS1 = """Kamu adalah analis substansi dokumen SROI yang bekerja dalam sebuah pipeline editorial.

== A. POSISI KERJA ==

Tugasmu bukan merangkum Bab 7. Tugasmu adalah mengidentifikasi substansi inti dokumen yang akan menjadi sumber bagi bab-bab lain. Elemen yang kamu hasilkan akan dipakai oleh penulis Bab 1 (pendahuluan), Bab 3 (metodologi), Bab 4 (kondisi awal), Bab 8 (pembelajaran), dan Bab 9 (penutup). Setiap elemen harus bisa hidup di luar Bab 7 dan tetap bermakna.

== B. PRINSIP SELEKSI DAN GROUPING ==

Pilih elemen yang menjadi urat saraf dokumen, bukan yang sekadar mengisi halaman.

KAPAN MENGGABUNGKAN BLOCKS:
- Beberapa blocks yang membangun satu unit makna harus digabung menjadi satu elemen
- Contoh: callout_warning status data + paragraph angka + tabel rincian = satu elemen investasi, bukan tiga elemen terpisah
- Contoh: paragraph observed outcome + paragraph proxy outcome = dua elemen terpisah, karena berbeda level evidensinya

KAPAN MEMISAHKAN:
- Pisahkan jika dua blocks membawa klaim dengan level evidence yang berbeda (observed vs proxy)
- Pisahkan jika dua blocks akan dipakai oleh bab yang berbeda secara fungsi

KAPAN MELEWATI:
- Blocks yang hanya memperkuat angka yang sudah ditangkap elemen lain
- Blocks yang hanya relevan lokal di Bab 7 dan tidak akan dibawa ke bab lain
- Non-content blocks (divider, spacer, divider_thick) tidak boleh menjadi elemen mandiri

TANDA ELEMEN YANG TERLALU DANGKAL (hindari):
- Label hanya menyalin heading subjudul (contoh: "investment_section")
- Summary hanya mengulang angka tanpa konteks epistemik
- Satu block → satu elemen tanpa pertimbangan grouping
- Elemen yang isinya hanya bisa dimengerti dalam konteks Bab 7

== C. TIGA FIELD KRITIS — BEDA FUNGSI, BEDA ISI ==

1. element_type — KATEGORI ONTOLOGIS elemen.
Harus salah satu dari:
evaluative_mandate, program_positioning, scope_definition, stakeholder_structure,
investment_structure, output_structure, outcome_structure, adjustment_logic,
monetization_logic, evaluative_metric, interpretive_finding, learning_signal,
problem_signal, ideal_state_signal, strategy_signal, program_structure, other

2. use_affordances — FUNGSI LINTAS BAB elemen ini.
Tanya: "Di bab mana saja elemen ini relevan dipakai, dan dalam kapasitas apa?"
Pilih SEMUA yang berlaku, bukan hanya yang paling dekat.
Elemen penting harus punya minimal 3–5 affordances.
Jika elemen hanya punya 1 affordance, curigai apakah elemen itu terlalu sempit.

Affordances yang tersedia:
opening_mandate — cocok jadi mandat pembuka Bab 1
program_positioning — menjelaskan posisi program dalam ekosistem
scope_definition — mendefinisikan ruang lingkup evaluasi
methodological_reference — mendukung penjelasan metode di Bab 3
problem_justification — mendukung argumen masalah di Bab 4
baseline_context — memberi konteks kondisi awal
ideal_state_anchor — memberi gambaran kondisi ideal
strategy_alignment — mendukung narasi strategi
implementation_reference — merujuk pada pelaksanaan program
outcome_reference — merujuk pada hasil/dampak program
adjustment_reference — merujuk pada proses penyesuaian (DDAT, dll)
monetization_reference — merujuk pada monetisasi nilai sosial
learning_anchor — cocok untuk pembelajaran di Bab 8
closing_summary — cocok untuk sintesis penutup Bab 9
recommendation_basis — menjadi dasar rekomendasi
visual_candidate — cocok divisualisasikan sebagai tabel/grafik

Extension boleh dengan prefix x_ (contoh: x_inclusion_signal)

JANGAN gunakan nilai dari element_type sebagai use_affordances.
SALAH: use_affordances = ["investment_structure"] → ini element_type
BENAR: use_affordances = ["opening_mandate", "methodological_reference", "closing_summary"]

3. guardrail_notes — LARANGAN DAN SYARAT PEMAKAIAN DOWNSTREAM.
Tulis sebagai instruksi operasional untuk penulis bab lain.
Minimal 1–3 guardrail per elemen yang material.
Wajib kosong HANYA untuk elemen dengan materiality sangat rendah.

FORMAT YANG BENAR (operasional, melarang, atau mensyaratkan):
✓ "Jangan tampilkan angka ini sebagai headline tanpa menyebut komposisi observed vs proxy."
✓ "Jika dibawa ke Bab I, perlakukan sebagai mandat evaluatif, bukan kesimpulan instan."
✓ "Jangan gabungkan elemen ini dengan outcome proxy tanpa membedakan level evidensinya."

FORMAT YANG SALAH (aspiratif, generik):
✗ "Ensure accuracy of data."
✗ "Verify alignment with program objectives."

== D. EVIDENCE RUBRIC — PILIH DENGAN SADAR ==

final   = data dari transaksi aktual, dokumen keuangan terverifikasi, atau pengukuran langsung
proxy   = estimasi berdasarkan referensi kebijakan, benchmark, atau proxy value — BUKAN data langsung
inferred = ditarik dari pola atau implikasi logis, belum dikonfirmasi data apapun
mixed   = campuran: sebagian blocks punya evidence final, sebagian proxy atau inferred
pending = data ada di sumber tapi belum diverifikasi pada saat ekstraksi ini

ATURAN PESSIMISTIC INFERENCE:
Jika provenance elemen mencakup blocks dengan evidence yang berbeda, gunakan yang paling lemah.
Contoh: satu block "final" + satu block bertanda "under confirmation" → evidence_status = "mixed"
Jangan memaksa "final" jika ada sinyal kehati-hatian di blocks sumber.

== E. CONTOH OUTPUT YANG DITOLAK ==

Contoh 1 — terlalu atomik, guardrail kosong, affordance sempit:
{
  "label": "investment_summary",
  "use_affordances": ["implementation_reference"],
  "guardrail_notes": [],
  "evidence_status": "final"
}
DITOLAK karena: satu block, satu affordance, guardrail kosong, tidak menangkap sinyal under confirmation.

Contoh 2 — label hanya menyalin heading:
{
  "label": "section_74_investasi",
  "summary": "Bagian ini membahas investasi program."
}
DITOLAK karena: label tidak substantif, summary tidak abstraktif.

== OUTPUT FORMAT ==

Output HANYA JSON array of elements (bungkus dalam object dengan key 'elements').
Tidak ada teks lain di luar JSON.

{
  "label": "snake_case_substantif",
  "element_type": "...",
  "element_type_note": null,
  "summary": "1-3 kalimat abstraktif yang bisa berdiri sendiri di luar Bab 7",
  "source_block_refs": ["..."],
  "source_block_fingerprints": ["..."],
  "source_block_types": ["..."],
  "evidence_status": "final|proxy|inferred|mixed|pending",
  "use_affordances": ["min 2, idealnya 3-5 untuk elemen material"],
  "guardrail_notes": ["larangan atau syarat operasional, bukan aspirasi"]
}"""

SYSTEM_PASS2 = """Kamu adalah analis SROI yang mengevaluasi hierarki kepentingan elemen substansi dan menyusun pagar penggunaan downstream.

== A. FUNGSI PASS INI ==

Pass ini memiliki tiga tugas yang berbeda:
1. Menetapkan scoring — hierarki kepentingan elemen dalam dokumen
2. Menulis guardrail_notes per elemen — larangan pemakaian spesifik (dimasukkan ke registry)
3. Menyusun guardrails packet — aturan global dan per-elemen untuk dokumen

== B. SCORING ANCHORS (konteks SROI) ==

materiality_score — seberapa penting elemen ini untuk klaim evaluatif dokumen:
5 = elemen episentrum: klaim SROI inti, struktur outcome, basis investasi utama
4 = elemen penting: mendukung klaim inti, perlu di hampir semua bab
3 = elemen kontekstual: relevan tapi bukan penentu narasi
2 = elemen pendukung: detail lokal yang kadang diperlukan
1 = elemen minor: informatif tapi bisa diabaikan

reusability_score — seberapa banyak bab yang akan memakai elemen ini:
5 = dipakai di 5+ bab dengan fungsi berbeda
4 = dipakai di 3–4 bab
3 = dipakai di 2 bab
2 = hanya relevan di 1–2 konteks terbatas
1 = hanya relevan di Bab 7

priority = "high" jika sum >= 8, "medium" jika 5–7, "low" jika <= 4

CATATAN: Elemen SROI metric, outcome structure, dan investment basis hampir selalu 4–5.
Jangan memberi skor medium untuk elemen yang jelas menjadi tesis evaluatif dokumen.

== C. GUARDRAIL_NOTES PER ELEMEN (masuk ke registry) ==

Untuk setiap elemen, tulis 1–3 guardrail_notes dalam format operasional.
Ini berbeda dari element_guardrails — guardrail_notes adalah instruksi ringkas
yang akan dibaca langsung oleh penulis bab saat menggunakan elemen ini.

Format: kalimat larangan atau syarat, bukan aspirasi.
Minimal untuk elemen dengan materiality >= 4.
Boleh kosong untuk elemen materiality <= 2.

== D. GUARDRAILS PACKET ==

global_guardrails: 1–3 aturan yang berlaku untuk seluruh dokumen.
Tulis sebagai larangan atau prasyarat, bukan pernyataan umum.
Contoh BENAR: "Jangan gunakan angka SROI tanpa menyebut komposisi observed vs proxy yang menopangnya."
Contoh SALAH: "Ensure all elements are aligned with program objectives."

element_guardrails: 1 guardrail per elemen material (materiality >= 3).
Fokus pada risiko pemakaian yang paling mungkin terjadi di downstream bab.
applies_to: gunakan label elemen (bukan SUB-xxx) — akan diresolve oleh sistem.

== OUTPUT FORMAT ==

Output HANYA JSON object:
{
  "scored_elements": [
    {
      "temp_element_key": "TMP-001",
      "label": "label_dari_pass1",
      "materiality_score": 1-5,
      "reusability_score": 1-5,
      "guardrail_notes": ["larangan operasional 1", "larangan operasional 2"]
    }
  ],
  "global_guardrails": [
    {
      "guardrail_id": "SG-001",
      "scope": "document",
      "applies_to": "all",
      "rule": "larangan atau prasyarat operasional",
      "severity": "low|medium|high|critical"
    }
  ],
  "element_guardrails": [
    {
      "guardrail_id": "SG-E-001",
      "scope": "element",
      "applies_to": "label_elemen",
      "rule": "larangan atau prasyarat operasional untuk elemen ini",
      "severity": "low|medium|high|critical"
    }
  ]
}"""


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────

def validate_affordances(affordances: list) -> list[str]:
    errors = []
    for aff in affordances:
        if aff not in CORE_AFFORDANCES and not RE_AFFORDANCE_EXT.match(aff):
            errors.append(f"affordance tidak valid: '{aff}'")
    return errors


def validate_element(
    el: dict,
    idx: int,
    non_content_types: set[str],
) -> list[str]:
    errors = []
    required = [
        "element_id", "label", "element_type", "summary",
        "source_block_refs", "source_block_fingerprints", "source_block_types",
        "evidence_status", "use_affordances", "guardrail_notes",
        "materiality_score", "reusability_score", "priority", "status",
    ]
    for f in required:
        if f not in el:
            errors.append(f"elemen[{idx}] ({el.get('label','?')}): field '{f}' wajib ada")

    etype = el.get("element_type")
    if etype and etype not in VALID_ELEMENT_TYPES:
        errors.append(f"elemen[{idx}]: element_type '{etype}' tidak valid")
    if etype == "other" and not el.get("element_type_note"):
        errors.append(f"elemen[{idx}]: element_type=other wajib punya element_type_note")
    if etype != "other" and el.get("element_type_note") is not None:
        errors.append(f"elemen[{idx}]: element_type_note harus null jika bukan 'other'")

    evst = el.get("evidence_status")
    if evst and evst not in VALID_EVIDENCE_STATUSES:
        errors.append(f"elemen[{idx}]: evidence_status '{evst}' tidak valid")

    # Provenance strict length equality
    refs  = el.get("source_block_refs", [])
    fps   = el.get("source_block_fingerprints", [])
    types = el.get("source_block_types", [])
    if not (len(refs) == len(fps) == len(types)):
        errors.append(
            f"elemen[{idx}]: source arrays harus panjang sama "
            f"(refs={len(refs)}, fps={len(fps)}, types={len(types)})"
        )
    if len(refs) == 0:
        errors.append(f"elemen[{idx}]: source arrays tidak boleh kosong")

    # R-A2-07: semua source adalah non-content → fatal
    # Mixed-source (sebagian non-content) masih diizinkan per R-A2-08
    non_content_hits = [t for t in types if t in non_content_types]
    if non_content_hits and len(non_content_hits) == len(types):
        errors.append(
            f"elemen[{idx}] ({el.get('label','?')}): "
            f"seluruh source_block_types adalah non-content blocks "
            f"({', '.join(non_content_hits)}). "
            f"Non-content blocks tidak boleh menjadi standalone element (R-A2-07)."
        )

    aff_errors = validate_affordances(el.get("use_affordances", []))
    errors.extend([f"elemen[{idx}]: {e}" for e in aff_errors])
    if not el.get("use_affordances"):
        errors.append(f"elemen[{idx}]: use_affordances tidak boleh kosong")

    mat = el.get("materiality_score")
    reu = el.get("reusability_score")
    if mat is not None and (not isinstance(mat, int) or mat < 1 or mat > 5):
        errors.append(f"elemen[{idx}]: materiality_score harus integer 1-5")
    if reu is not None and (not isinstance(reu, int) or reu < 1 or reu > 5):
        errors.append(f"elemen[{idx}]: reusability_score harus integer 1-5")

    # Priority consistency
    if mat and reu:
        total = mat + reu
        expected = "high" if total >= 8 else "medium" if total >= 5 else "low"
        if el.get("priority") != expected:
            errors.append(
                f"elemen[{idx}]: priority '{el.get('priority')}' tidak konsisten "
                f"dengan skor ({mat}+{reu}={total}, expected='{expected}')"
            )

    return errors


def semantic_lint(elements: list[dict]) -> list[dict]:
    """
    Semantic lint — deteksi pola suspicious yang lolos schema validation.
    Returns list of warning dicts. Tidak fatal, tapi harus di-review manual.

    Pola yang dicek:
    - Semua evidence_status = "final" (possible over-confidence)
    - Semua guardrail_notes = [] (incomplete specification)
    - Mayoritas elemen punya 1 affordance (possible under-specification)
    - Guardrails mengandung pola aspiratif "Ensure.*" atau "Verify.*"
    - Elemen high materiality tanpa guardrail_notes
    """
    warnings = []
    if not elements:
        return warnings

    # 1. Semua evidence_status final
    all_final = all(el.get("evidence_status") == "final" for el in elements)
    if all_final and len(elements) > 3:
        warnings.append({
            "code": "LINT_ALL_EVIDENCE_FINAL",
            "message": (
                f"Semua {len(elements)} elemen punya evidence_status='final'. "
                "Periksa apakah ada elemen yang seharusnya 'proxy', 'mixed', atau 'inferred'."
            ),
        })

    # 2. Semua guardrail_notes kosong
    all_empty_notes = all(not el.get("guardrail_notes") for el in elements)
    if all_empty_notes:
        warnings.append({
            "code": "LINT_ALL_GUARDRAIL_NOTES_EMPTY",
            "message": "Semua elemen punya guardrail_notes kosong. Pass 1 dan Pass 2 tidak mengisi field ini.",
        })

    # 3. Mayoritas affordance = 1
    single_aff = sum(1 for el in elements if len(el.get("use_affordances", [])) <= 1)
    if single_aff / len(elements) > 0.6:
        warnings.append({
            "code": "LINT_AFFORDANCES_UNDERSPECIFIED",
            "message": (
                f"{single_aff}/{len(elements)} elemen hanya punya 1 affordance. "
                "Elemen material biasanya punya 3–5 affordances."
            ),
        })

    # 4. Grouping quality — Q_group: deteksi atomistic extraction drift
    # Elemen material (materiality >= 4) yang hanya berasal dari 1 block
    # menunjukkan bahwa abstraction bergeser kembali ke extraction.
    # Denominator = material_elements (bukan len(elements)) agar proporsi akurat.
    if any(el.get("materiality_score", 0) is not None for el in elements):
        single_block_material = sum(
            1 for el in elements
            if len(el.get("source_block_refs", [])) == 1
            and el.get("materiality_score", 0) >= 4
        )
        material_elements = sum(1 for el in elements if el.get("materiality_score", 0) >= 4)
        if material_elements > 0 and (single_block_material / material_elements) > 0.6:
            drift_ratio = single_block_material / material_elements
            severity = "high" if drift_ratio >= 0.8 else "medium"
            warnings.append({
                "code": "LINT_ATOMIC_EXTRACTION_DRIFT",
                "severity": severity,
                "message": (
                    f"{single_block_material}/{material_elements} elemen material "
                    f"({drift_ratio:.0%}) hanya berasal dari 1 block. "
                    f"Mayoritas elemen material seharusnya menggabungkan beberapa blocks. "
                    f"Periksa apakah abstraction berubah menjadi extraction."
                ),
            })

    # 4. High materiality tanpa guardrail_notes
    import re as _re
    aspirational_pattern = _re.compile(r'^(Ensure|Verify|Make sure|Check|Confirm)', _re.IGNORECASE)
    for el in elements:
        mat = el.get("materiality_score", 0)
        notes = el.get("guardrail_notes", [])
        if mat >= 4 and not notes:
            warnings.append({
                "code": "LINT_HIGH_MATERIALITY_NO_GUARDRAIL",
                "element_id": el.get("element_id", "?"),
                "message": (
                    f"{el.get('element_id','?')} ({el.get('label','?')}): "
                    f"materiality={mat} tapi guardrail_notes kosong."
                ),
            })
        # 5. Guardrail aspiratif
        for note in notes:
            if aspirational_pattern.match(note):
                warnings.append({
                    "code": "LINT_ASPIRATIONAL_GUARDRAIL",
                    "element_id": el.get("element_id", "?"),
                    "message": (
                        f"{el.get('element_id','?')}: guardrail_note terdengar aspiratif: '{note[:80]}'"
                    ),
                })

    return warnings


def validate_provenance_against_packet(
    elements: list[dict],
    block_identity_packet: dict,
) -> list[str]:
    """Pastikan semua source_block_refs dan fingerprints ada di block_identity_packet."""
    all_refs: dict[str, bool] = {}
    all_fps:  dict[str, bool] = {}
    for ch in block_identity_packet.get("chapters", []):
        for b in ch.get("blocks", []):
            all_refs[b["block_ref"]]         = True
            all_fps[b["block_fingerprint"]]  = True

    errors = []
    for i, el in enumerate(elements):
        for ref in el.get("source_block_refs", []):
            if ref not in all_refs:
                errors.append(
                    f"elemen[{i}] ({el.get('label','?')}): "
                    f"source_block_ref '{ref}' tidak ada di block_identity_packet"
                )
        for fp in el.get("source_block_fingerprints", []):
            if fp not in all_fps:
                errors.append(
                    f"elemen[{i}] ({el.get('label','?')}): "
                    f"source_block_fingerprint '{fp}' tidak ada di block_identity_packet"
                )
    return errors


# ─────────────────────────────────────────────
# LLM CALLS
# ─────────────────────────────────────────────

def call_llm(client: OpenAI, system: str, user: str, label: str) -> str:
    """Single LLM call, returns raw text content."""
    print(f"[A2] LLM call: {label} ...", flush=True)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def parse_json_response(raw: str, label: str) -> tuple[Any, str | None]:
    """Parse JSON dari LLM response. Returns (parsed, error_msg)."""
    try:
        # response_format=json_object menjamin valid JSON
        # tapi kadang model wrap dalam object dengan key random
        data = json.loads(raw)
        return data, None
    except json.JSONDecodeError as e:
        return None, f"{label}: JSON parse error — {e}"


def build_pass1_prompt(
    chapter_id: str,
    blocks: list[dict],
    non_content_types: set[str],
) -> str:
    """Bentuk user prompt untuk Pass 1."""
    # Filter non-content blocks dari context (masih disertakan tapi diberi label)
    block_lines = []
    for b in blocks:
        is_nc = b.get("type", "") in non_content_types
        note  = " [NON-CONTENT — jangan jadikan elemen mandiri]" if is_nc else ""
        block_lines.append(
            f"block_ref: {b.get('block_ref','?')}\n"
            f"block_fingerprint: {b.get('block_fingerprint','?')}\n"
            f"type: {b.get('type','?')}{note}\n"
            f"text: {b.get('text') or '[kosong]'}\n"
        )

    blocks_text = "\n---\n".join(block_lines)
    return (
        f"Chapter sumber substansi: {chapter_id}\n\n"
        f"Blocks:\n\n{blocks_text}\n\n"
        f"Ekstrak elemen-elemen substantif yang material dan reusable dari blocks di atas.\n"
        f"Setiap elemen WAJIB menyertakan field 'temp_element_key' dengan format 'TMP-001', 'TMP-002', dst. "
        f"Key ini digunakan untuk menghubungkan Pass 1 dan Pass 2 secara stabil — lebih reliable dari label.\n"
        f"Output HANYA JSON array (bungkus dalam object dengan key 'elements')."
    )


def build_pass2_prompt(pass1_elements: list[dict], document_context: str) -> str:
    """Bentuk user prompt untuk Pass 2."""
    elements_json = json.dumps(pass1_elements, ensure_ascii=False, indent=2)
    return (
        f"Konteks dokumen: {document_context}\n\n"
        f"Elemen hasil ekstraksi Pass 1:\n{elements_json}\n\n"
        f"Berikan scoring untuk setiap elemen. Gunakan field 'temp_element_key' (TMP-001, TMP-002, dst.) "
        f"sebagai identifier di scored_elements — BUKAN label. "
        f"Ini memastikan scoring tidak hilang meski ada perbedaan kecil di label.\n"
        f"Sertakan juga 'guardrail_notes' per elemen di scored_elements.\n"
        f"Susun global_guardrails dan element_guardrails (applies_to menggunakan label elemen).\n"
        f"Output HANYA JSON object sesuai format yang diberikan."
    )


# ─────────────────────────────────────────────
# ASSEMBLY
# ─────────────────────────────────────────────

def assemble_registry(
    pass1_elements: list[dict],
    pass2_data: dict,
    document_id: str,
    program_code: str,
    substance_chapter_id: str,
    substance_mode: str,
    substance_basis: str,
) -> tuple[dict, dict]:
    """
    Gabungkan hasil Pass 1 + Pass 2 menjadi substance_registry dan substance_guardrails.
    Returns (registry, guardrails).
    """
    # Index scored elements: primary via temp_element_key, fallback via label
    # Ini mencegah scoring hilang silent saat LLM typo label di Pass 2
    scored_by_key   = {}
    scored_by_label = {}
    for e in pass2_data.get("scored_elements", []):
        key   = e.get("temp_element_key", "")
        label = e.get("label", "")
        if key:
            scored_by_key[key] = e
        if label:
            scored_by_label[label] = e

    now = datetime.now(timezone.utc).isoformat()
    miss_count = 0

    elements = []
    for i, el in enumerate(pass1_elements):
        label = el.get("label", f"element_{i}")
        key   = el.get("temp_element_key", f"TMP-{(i+1):03d}")

        # Resolve scoring: key primary, label fallback
        score = scored_by_key.get(key) or scored_by_label.get(label)
        if score is None:
            miss_count += 1
            score = {}

        mat   = int(score.get("materiality_score", 3))
        reu   = int(score.get("reusability_score", 3))
        total = mat + reu
        pri   = "high" if total >= 8 else "medium" if total >= 5 else "low"

        # Merge guardrail_notes: Pass 1 (dari extraction) + Pass 2 (dari evaluation)
        # Pass 2 punya konteks scoring sehingga guardrailnya lebih terarah
        pass1_notes = el.get("guardrail_notes", []) or []
        pass2_notes = score.get("guardrail_notes", []) or []
        # Deduplikasi — prioritaskan Pass 2, tambah Pass 1 yang tidak overlap
        merged_notes = list(pass2_notes)
        for note in pass1_notes:
            if note not in merged_notes:
                merged_notes.append(note)

        elements.append({
            "element_id":          f"SUB-{(i+1):03d}",
            "label":               label,
            "element_type":        el.get("element_type", "other"),
            "element_type_note":   el.get("element_type_note"),
            "summary":             el.get("summary", ""),
            "source_block_refs":   el.get("source_block_refs", []),
            "source_block_fingerprints": el.get("source_block_fingerprints", []),
            "source_block_types":  el.get("source_block_types", []),
            "evidence_status":     el.get("evidence_status", "inferred"),
            "use_affordances":     el.get("use_affordances", []),
            "guardrail_notes":     merged_notes,
            "materiality_score":   mat,
            "reusability_score":   reu,
            "priority":            pri,
            "status":              "active",
        })

    # Summary stats
    by_type     = {}
    by_evidence = {}
    by_priority = {}
    for el in elements:
        by_type[el["element_type"]]        = by_type.get(el["element_type"], 0) + 1
        by_evidence[el["evidence_status"]] = by_evidence.get(el["evidence_status"], 0) + 1
        by_priority[el["priority"]]        = by_priority.get(el["priority"], 0) + 1

    registry = {
        "activity":             ACTIVITY,
        "document_id":          document_id,
        "program_code":         program_code,
        "substance_mode":       substance_mode,
        "substance_chapter_id": substance_chapter_id,
        "substance_basis":      substance_basis,
        "generated_at":         now,
        "pipeline_version":     PIPELINE_VERSION,
        "elements":             elements,
        "summary": {
            "total_elements": len(elements),
            "by_type":        by_type,
            "by_evidence_status": by_evidence,
            "by_priority":    by_priority,
        },
        "warnings":  [],
        "failures":  [],
    }

    # Warning jika ada elemen yang scoring-nya tidak ter-resolve
    if miss_count > 0:
        registry["warnings"].append({
            "level": "warning",
            "code": "SCORING_KEY_MISS",
            "message": (
                f"{miss_count} elemen tidak ter-resolve oleh Pass 2 scoring "
                f"(baik via temp_element_key maupun label). "
                f"Skor default digunakan (materiality=3, reusability=3)."
            ),
        })

    # Guardrails: re-number IDs secara berurutan
    global_guardrails  = []
    element_guardrails = []
    g_seq = 1
    e_seq = 1

    for g in pass2_data.get("global_guardrails", []):
        global_guardrails.append({
            "guardrail_id": f"SG-{g_seq:03d}",
            "scope":       g.get("scope", "document"),
            "applies_to":  g.get("applies_to", "all"),
            "rule":        g.get("rule", ""),
            "severity":    g.get("severity", "medium"),
        })
        g_seq += 1

    # Map element guardrails: resolve applies_to dari label ke element_id
    label_to_id = {el["label"]: el["element_id"] for el in elements}
    for eg in pass2_data.get("element_guardrails", []):
        applies_raw = eg.get("applies_to", "")
        # applies_to bisa berupa label atau SUB-xxx
        resolved = label_to_id.get(applies_raw, applies_raw)
        element_guardrails.append({
            "guardrail_id": f"SG-E-{e_seq:03d}",
            "scope":       "element",
            "applies_to":  resolved,
            "rule":        eg.get("rule", ""),
            "severity":    eg.get("severity", "medium"),
        })
        e_seq += 1

    guardrails = {
        "activity":             ACTIVITY,
        "document_id":          document_id,
        "program_code":         program_code,
        "substance_chapter_id": substance_chapter_id,
        "generated_at":         now,
        "pipeline_version":     PIPELINE_VERSION,
        "global_guardrails":    global_guardrails,
        "element_guardrails":   element_guardrails,
    }

    return registry, guardrails


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────

def run(
    input_data: dict,
    api_key: str | None = None,
    dry_run: bool = False,
) -> tuple[dict, dict]:
    """
    Jalankan A2 Substance Extractor.

    Input dict harus mengandung:
      - pipeline_control_block
      - block_identity_packet  (output A1)

    Returns (substance_registry, substance_guardrails).
    """
    pcb     = input_data.get("pipeline_control_block", {})
    bip     = input_data.get("block_identity_packet", {})
    doc_id  = pcb.get("document_id", "unknown")
    prog    = pcb.get("program_code", "unknown")
    mode    = pcb.get("substance_mode", "confirmed")
    basis   = pcb.get("substance_basis", "unknown")
    nct     = set(pcb.get("non_content_block_types", list(NON_CONTENT_BLOCK_TYPES_DEFAULT)))

    # ── Step 1: Resolve source chapter ────────
    substance_chapter_id = pcb.get("substance_chapter_id")
    if mode == "confirmed" and not substance_chapter_id:
        raise ValueError("substance_mode=confirmed membutuhkan substance_chapter_id")

    # ── Step 2: Validate source chapter ───────
    chapters = {ch["chapter_id"]: ch for ch in bip.get("chapters", [])}
    if substance_chapter_id and substance_chapter_id not in chapters:
        raise ValueError(
            f"substance_chapter_id '{substance_chapter_id}' "
            f"tidak ditemukan di block_identity_packet. "
            f"Tersedia: {list(chapters.keys())}"
        )

    # Jika discovery mode tanpa chapter_id, pakai chapter pertama
    if not substance_chapter_id:
        substance_chapter_id = next(iter(chapters))
        print(f"[A2] discovery mode — menggunakan chapter pertama: {substance_chapter_id}")

    source_chapter = chapters[substance_chapter_id]
    blocks         = source_chapter.get("blocks", [])

    print(f"[A2] Source chapter : {substance_chapter_id}")
    print(f"[A2] Total blocks   : {len(blocks)}")

    # ── Dry run — kembalikan skeleton kosong ──
    if dry_run:
        print("[A2] DRY RUN — skip LLM calls")
        registry, guardrails = assemble_registry(
            [], {}, doc_id, prog,
            substance_chapter_id, mode, basis,
        )
        registry["summary"]["warning_count"] = 0
        registry["summary"]["failure_count"] = 0
        return registry, guardrails

    # ── Step 3–8: Pass 1 — Extraction ─────────
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    prompt1 = build_pass1_prompt(substance_chapter_id, blocks, nct)
    raw1    = call_llm(client, SYSTEM_PASS1, prompt1, "Pass 1 — Extraction")
    data1, err1 = parse_json_response(raw1, "Pass 1")
    if err1:
        raise RuntimeError(f"Pass 1 parse error: {err1}\nRaw: {raw1[:500]}")

    # Normalise: model mungkin return {"elements": [...]} atau langsung [...]
    if isinstance(data1, list):
        pass1_elements = data1
    elif isinstance(data1, dict):
        pass1_elements = data1.get("elements", [])
    else:
        raise RuntimeError(f"Pass 1 unexpected type: {type(data1)}")

    # Inject temp_element_key deterministik jika LLM tidak mengisinya
    # Ini menjamin Pass 2 punya identifier stabil — tidak bergantung pada label
    for i, el in enumerate(pass1_elements):
        if not el.get("temp_element_key"):
            el["temp_element_key"] = f"TMP-{(i+1):03d}"

    print(f"[A2] Pass 1 complete : {len(pass1_elements)} elemen diekstrak")

    # ── Step 9: Pass 2 — Scoring + Guardrails ─
    doc_context = (
        f"Dokumen: {doc_id} | Program: {prog} | "
        f"Chapter substansi: {substance_chapter_id} | Basis: {basis}"
    )
    prompt2 = build_pass2_prompt(pass1_elements, doc_context)
    raw2    = call_llm(client, SYSTEM_PASS2, prompt2, "Pass 2 — Scoring")
    data2, err2 = parse_json_response(raw2, "Pass 2")
    if err2:
        raise RuntimeError(f"Pass 2 parse error: {err2}\nRaw: {raw2[:500]}")

    if not isinstance(data2, dict):
        raise RuntimeError(f"Pass 2 unexpected type: {type(data2)}")

    print(f"[A2] Pass 2 complete : "
          f"{len(data2.get('scored_elements',[]))} scored, "
          f"{len(data2.get('global_guardrails',[]))} global guardrails, "
          f"{len(data2.get('element_guardrails',[]))} element guardrails")

    # ── Step 10: Assemble + Validate ──────────
    registry, guardrails = assemble_registry(
        pass1_elements, data2,
        doc_id, prog, substance_chapter_id, mode, basis,
    )

    # Validate elements
    all_errors = []
    for i, el in enumerate(registry["elements"]):
        all_errors.extend(validate_element(el, i, nct))

    # Validate provenance against block_identity_packet
    prov_errors = validate_provenance_against_packet(registry["elements"], bip)
    all_errors.extend(prov_errors)

    # R-A2-08: mixed-source warning (sebagian non-content) — tidak fatal
    for el in registry["elements"]:
        types = el.get("source_block_types", [])
        non_content_hits = [t for t in types if t in nct]
        if non_content_hits and len(non_content_hits) < len(types):
            registry["warnings"].append({
                "level": "warning",
                "code": "MIXED_SOURCE_NON_CONTENT",
                "element_id": el.get("element_id", "?"),
                "message": (
                    f"{el.get('element_id','?')} ({el.get('label','?')}): "
                    f"sebagian source adalah non-content blocks "
                    f"({', '.join(non_content_hits)}). "
                    f"Diizinkan, tetapi perlu review."
                ),
            })

    # Semantic lint — deteksi pola suspicious (warning, tidak fatal)
    lint_warnings = semantic_lint(registry["elements"])
    for lw in lint_warnings:
        registry["warnings"].append({
            "level": "warning",
            **lw,
        })
    if lint_warnings:
        print(f"[A2] Semantic lint   : {len(lint_warnings)} suspicious pattern(s)")

    if all_errors:
        registry["failures"] = [
            {"level": "fatal", "code": "VALIDATION_ERROR", "message": e}
            for e in all_errors
        ]
        registry["summary"]["failure_count"] = len(all_errors)
        registry["summary"]["warning_count"]  = len(registry["warnings"])
        print(f"[A2] VALIDATION FAILURES: {len(all_errors)}", file=sys.stderr)
        for e in all_errors:
            print(f"  {e}", file=sys.stderr)
    else:
        registry["summary"]["failure_count"] = 0
        registry["summary"]["warning_count"]  = len(registry["warnings"])
        print(f"[A2] Validation OK : {len(registry['elements'])} elemen valid")

    return registry, guardrails


# ─────────────────────────────────────────────
# CLI ENTRYPOINT
# ─────────────────────────────────────────────

def main():
    """
    Usage:
      python a2_substance_extractor.py <input_file> [registry_out] [guardrails_out]

    input_file   : JSON dengan keys pipeline_control_block + block_identity_packet
    registry_out : path output registry  (default: data/output/substance_registry.json)
    guardrails_out: path output guardrails (default: data/output/substance_guardrails.json)

    Env:
      OPENAI_API_KEY : required
      A2_DRY_RUN     : set ke "1" untuk skip LLM calls
    """
    if len(sys.argv) < 2:
        print(
            "Usage: python a2_substance_extractor.py <input_file> "
            "[registry_out] [guardrails_out]",
            file=sys.stderr,
        )
        sys.exit(1)

    input_path     = Path(sys.argv[1])
    registry_path  = Path(sys.argv[2]) if len(sys.argv) > 2 \
                     else Path("data/output/substance_registry.json")
    guardrails_path = Path(sys.argv[3]) if len(sys.argv) > 3 \
                     else Path("data/output/substance_guardrails.json")

    if not input_path.exists():
        print(f"ERROR: Input file tidak ditemukan: {input_path}", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    dry_run = os.environ.get("A2_DRY_RUN", "0") == "1"

    if not api_key and not dry_run:
        print("ERROR: OPENAI_API_KEY tidak di-set. Set env atau gunakan A2_DRY_RUN=1",
              file=sys.stderr)
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        input_data = json.load(f)

    try:
        registry, guardrails = run(input_data, api_key=api_key, dry_run=dry_run)
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    guardrails_path.parent.mkdir(parents=True, exist_ok=True)

    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    with open(guardrails_path, "w", encoding="utf-8") as f:
        json.dump(guardrails, f, ensure_ascii=False, indent=2)

    summary = registry.get("summary", {})
    status  = "FAILED" if registry.get("failures") else "OK"
    print(f"\n[A2] Status          : {status}")
    print(f"[A2] Elements        : {summary.get('total_elements', 0)}")
    print(f"[A2] By priority     : {summary.get('by_priority', {})}")
    print(f"[A2] Registry out    : {registry_path}")
    print(f"[A2] Guardrails out  : {guardrails_path}")

    if registry.get("failures"):
        sys.exit(1)


if __name__ == "__main__":
    main()
