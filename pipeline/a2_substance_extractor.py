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

SYSTEM_PASS1 = """Kamu adalah analis dokumen SROI (Social Return on Investment) yang berpengalaman.
Tugasmu adalah membaca blocks dari chapter substansi sebuah laporan SROI dan mengekstrak elemen-elemen substantif yang material dan reusable untuk penulisan bab-bab lain.

PENTING:
- Ekstraksi SELEKTIF, bukan exhaustive copy. Pilih elemen yang benar-benar material.
- Setiap elemen harus mewakili satu unit makna yang koheren.
- Summary harus abstraktif (1-3 kalimat), bukan salinan mentah teks.
- Provenance harus akurat — gunakan block_ref, block_fingerprint, dan type persis dari input.
- Non-content blocks (divider, spacer, divider_thick) TIDAK boleh menjadi elemen mandiri.
- Evidence status mengikuti interpretasi teraman (weakest safe interpretation).
- use_affordances dari vocab berikut (boleh kombinasi): opening_mandate, program_positioning, scope_definition, methodological_reference, problem_justification, baseline_context, ideal_state_anchor, strategy_alignment, implementation_reference, outcome_reference, adjustment_reference, monetization_reference, learning_anchor, closing_summary, recommendation_basis, visual_candidate. Extension boleh dengan prefix x_.
- element_type harus salah satu dari: evaluative_mandate, program_positioning, scope_definition, stakeholder_structure, investment_structure, output_structure, outcome_structure, adjustment_logic, monetization_logic, evaluative_metric, interpretive_finding, learning_signal, problem_signal, ideal_state_signal, strategy_signal, program_structure, other.
- Jika element_type = other, wajib isi element_type_note.

Output HANYA JSON array of elements. Tidak ada teks lain. Format tiap elemen:
{
  "label": "snake_case_name",
  "element_type": "...",
  "element_type_note": null,
  "summary": "...",
  "source_block_refs": ["..."],
  "source_block_fingerprints": ["..."],
  "source_block_types": ["..."],
  "evidence_status": "final|proxy|pending|inferred|mixed",
  "use_affordances": ["..."],
  "guardrail_notes": ["..."]
}"""

SYSTEM_PASS2 = """Kamu adalah analis SROI yang mengevaluasi kualitas elemen substansi dokumen.
Tugasmu adalah memberi skor materiality dan reusability untuk setiap elemen, lalu menyusun guardrails global dan per-elemen.

Aturan scoring:
- materiality_score: 1-5 (1=minor, 5=inti dokumen)
- reusability_score: 1-5 (1=sempit, 5=sangat reusable lintas bab)
- priority: "high" jika sum>=8, "medium" jika 5-7, "low" jika <=4

Guardrail rules:
- global_guardrails: aturan yang berlaku seluruh dokumen (scope: document)
- element_guardrails: aturan spesifik per elemen (scope: element, applies_to: element_id)
- severity: low | medium | high | critical
- Guardrail harus spesifik terhadap risiko pemakaian, bukan generik.

Output HANYA JSON object. Format:
{
  "scored_elements": [
    {
      "label": "label_dari_pass1",
      "materiality_score": 1-5,
      "reusability_score": 1-5,
      "priority": "high|medium|low"
    }
  ],
  "global_guardrails": [
    {
      "guardrail_id": "SG-001",
      "scope": "document",
      "applies_to": "all",
      "rule": "...",
      "severity": "low|medium|high|critical"
    }
  ],
  "element_guardrails": [
    {
      "guardrail_id": "SG-E-001",
      "scope": "element",
      "applies_to": "SUB-001",
      "rule": "...",
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
        f"Ekstrak elemen-elemen substantif yang material dan reusable dari blocks di atas. "
        f"Output HANYA JSON array (bungkus dalam object dengan key 'elements')."
    )


def build_pass2_prompt(pass1_elements: list[dict], document_context: str) -> str:
    """Bentuk user prompt untuk Pass 2."""
    elements_json = json.dumps(pass1_elements, ensure_ascii=False, indent=2)
    return (
        f"Konteks dokumen: {document_context}\n\n"
        f"Elemen hasil ekstraksi Pass 1:\n{elements_json}\n\n"
        f"Berikan scoring (materiality_score, reusability_score, priority) "
        f"untuk setiap elemen, dan susun global_guardrails serta element_guardrails. "
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
    scored = {e["label"]: e for e in pass2_data.get("scored_elements", [])}
    now    = datetime.now(timezone.utc).isoformat()

    elements = []
    for i, el in enumerate(pass1_elements):
        label = el.get("label", f"element_{i}")
        score = scored.get(label, {})
        mat   = int(score.get("materiality_score", 3))
        reu   = int(score.get("reusability_score", 3))
        total = mat + reu
        pri   = "high" if total >= 8 else "medium" if total >= 5 else "low"

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
            "guardrail_notes":     el.get("guardrail_notes", []),
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
