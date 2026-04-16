"""
Sprint 2 — Official Tests: Non-Content Block Validation
File: tests/sprint_2/test_a2_non_content_validation.py

Mencakup:
  R-A2-07 — non-content blocks tidak boleh menjadi standalone substance elements
  R-A2-08 — mixed-source element (sebagian non-content) diperbolehkan, menghasilkan warning

Test resmi ini berfungsi sebagai acceptance gate untuk kedua rule tersebut.
Terpisah dari unit test suite agar mudah dirujuk saat review sprint.
"""

import sys
import os
import json
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.a2_substance_extractor import (
    validate_element,
    run,
)

# ─────────────────────────────────────────────
# SHARED FIXTURES
# ─────────────────────────────────────────────

NON_CONTENT_TYPES = {"divider", "divider_thick", "spacer"}

def make_pcb(**overrides):
    base = {
        "document_id": "esd_report_v1",
        "program_code": "ESD",
        "substance_mode": "confirmed",
        "substance_chapter_id": "bab_7",
        "substance_basis": "manual_audit",
        "non_content_block_types": list(NON_CONTENT_TYPES),
    }
    base.update(overrides)
    return base


def make_bip():
    """Block identity packet dengan blocks yang dibutuhkan semua test."""
    return {
        "activity": "block_id_injector",
        "document_id": "esd_report_v1",
        "program_code": "ESD",
        "generated_at": "2025-01-01T00:00:00+00:00",
        "chapters": [{
            "chapter_id": "bab_7",
            "chapter_type": "implementation_core",
            "blocks": [
                {
                    "block_ref": "bab_7.B001",
                    "block_fingerprint": "bab_7__paragraph_lead__implementasi_program",
                    "original_index": 0,
                    "type": "paragraph_lead",
                    "text": "Implementasi program dilakukan secara bertahap.",
                },
                {
                    "block_ref": "bab_7.B002",
                    "block_fingerprint": "bab_7__table_borderless__dimensi_program",
                    "original_index": 1,
                    "type": "table_borderless",
                    "headers": ["Dimensi", "Nilai"],
                },
                {
                    "block_ref": "bab_7.B003",
                    "block_fingerprint": "bab_7__divider__structural_2",
                    "original_index": 2,
                    "type": "divider",
                },
                {
                    "block_ref": "bab_7.B004",
                    "block_fingerprint": "bab_7__spacer__structural_3",
                    "original_index": 3,
                    "type": "spacer",
                },
            ],
        }],
        "summary": {},
        "warnings": [],
        "failures": [],
    }


def make_element(**overrides):
    """Elemen valid sebagai baseline."""
    base = {
        "element_id": "SUB-001",
        "label": "sroi_result",
        "element_type": "evaluative_metric",
        "element_type_note": None,
        "summary": "SROI blended program adalah 1:1.14.",
        "source_block_refs": ["bab_7.B001"],
        "source_block_fingerprints": ["bab_7__paragraph_lead__implementasi_program"],
        "source_block_types": ["paragraph_lead"],
        "evidence_status": "final",
        "use_affordances": ["opening_mandate", "closing_summary"],
        "guardrail_notes": ["Jangan disebutkan terlalu dini."],
        "materiality_score": 5,
        "reusability_score": 4,
        "priority": "high",
        "status": "active",
    }
    base.update(overrides)
    return base


def make_input_data():
    return {
        "pipeline_control_block": make_pcb(),
        "block_identity_packet":  make_bip(),
    }


def make_llm_pass2_response(label: str, mat: int = 4, reu: int = 4) -> str:
    total = mat + reu
    priority = "high" if total >= 8 else "medium" if total >= 5 else "low"
    return json.dumps({
        "scored_elements": [{"label": label, "materiality_score": mat, "reusability_score": reu}],
        "global_guardrails": [],
        "element_guardrails": [],
    })


def run_with_mock_llm(pass1_elements: list, mat: int = 4, reu: int = 4) -> tuple:
    """Helper: jalankan run() dengan LLM di-mock."""
    label = pass1_elements[0]["label"] if pass1_elements else "el"

    call_count = [0]
    def side_effect(**kwargs):
        call_count[0] += 1
        resp = MagicMock()
        choice = MagicMock()
        if call_count[0] == 1:
            choice.message.content = json.dumps({"elements": pass1_elements})
        else:
            choice.message.content = make_llm_pass2_response(label, mat, reu)
        resp.choices = [choice]
        return resp

    with patch("pipeline.a2_substance_extractor.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = side_effect
        MockOpenAI.return_value = mock_client
        return run(make_input_data(), api_key="test-key")


# ─────────────────────────────────────────────
# R-A2-07 — FATAL: ALL NON-CONTENT SOURCE
# ─────────────────────────────────────────────

def test_r_a2_07_all_non_content_source_rejected():
    """
    R-A2-07: elemen yang seluruh source_block_types-nya adalah non-content blocks
    harus ditolak dengan fatal validation error.

    Case: source_block_types = ["divider"] — single non-content block.
    """
    el = make_element(
        source_block_refs=["bab_7.B003"],
        source_block_fingerprints=["bab_7__divider__structural_2"],
        source_block_types=["divider"],
    )

    errors = validate_element(el, 0, NON_CONTENT_TYPES)

    assert errors, "Harus ada error untuk all-non-content source"
    assert any("R-A2-07" in e for e in errors), (
        "Pesan error harus menyebut R-A2-07"
    )
    assert any("standalone" in e.lower() or "non-content" in e.lower() for e in errors), (
        "Pesan error harus menjelaskan bahwa non-content blocks tidak boleh standalone"
    )


# ─────────────────────────────────────────────
# R-A2-07 — FATAL: MULTIPLE NON-CONTENT SOURCES
# ─────────────────────────────────────────────

def test_r_a2_07_multiple_non_content_sources_rejected():
    """
    R-A2-07: elemen dengan lebih dari satu non-content block sebagai satu-satunya
    source harus tetap ditolak — bukan hanya single divider.

    Case: source_block_types = ["divider", "spacer"].
    """
    el = make_element(
        source_block_refs=["bab_7.B003", "bab_7.B004"],
        source_block_fingerprints=[
            "bab_7__divider__structural_2",
            "bab_7__spacer__structural_3",
        ],
        source_block_types=["divider", "spacer"],
    )

    errors = validate_element(el, 0, NON_CONTENT_TYPES)

    assert errors, "Harus ada error untuk semua source non-content"
    assert any("R-A2-07" in e for e in errors)


# ─────────────────────────────────────────────
# R-A2-07 + R-A2-08 — ALLOWED: PARTIAL NON-CONTENT
# ─────────────────────────────────────────────

def test_r_a2_07_partial_non_content_allowed():
    """
    R-A2-08: elemen dengan source campuran (sebagian content, sebagian non-content)
    DIPERBOLEHKAN — bukan fatal error di validate_element.

    Case: source_block_types = ["paragraph_lead", "divider"].
    Non-content block hanya ikut dalam grouping, bukan satu-satunya basis.
    """
    el = make_element(
        source_block_refs=["bab_7.B001", "bab_7.B003"],
        source_block_fingerprints=[
            "bab_7__paragraph_lead__implementasi_program",
            "bab_7__divider__structural_2",
        ],
        source_block_types=["paragraph_lead", "divider"],
    )

    errors = validate_element(el, 0, NON_CONTENT_TYPES)

    r_a2_07_errors = [e for e in errors if "R-A2-07" in e]
    assert not r_a2_07_errors, (
        f"Mixed source tidak boleh menghasilkan R-A2-07 error, "
        f"tapi dapat: {r_a2_07_errors}"
    )


# ─────────────────────────────────────────────
# R-A2-07 — UNAFFECTED: NORMAL SOURCE
# ─────────────────────────────────────────────

def test_r_a2_07_normal_source_unaffected():
    """
    Elemen dengan source_block_types yang semuanya content blocks
    tidak boleh terpengaruh oleh R-A2-07 validation.

    Case: source_block_types = ["paragraph_lead", "table_borderless"] — semua content.
    """
    el = make_element(
        source_block_refs=["bab_7.B001", "bab_7.B002"],
        source_block_fingerprints=[
            "bab_7__paragraph_lead__implementasi_program",
            "bab_7__table_borderless__dimensi_program",
        ],
        source_block_types=["paragraph_lead", "table_borderless"],
        materiality_score=4,
        reusability_score=4,
        priority="high",
    )

    errors = validate_element(el, 0, NON_CONTENT_TYPES)

    r_a2_07_errors = [e for e in errors if "R-A2-07" in e]
    assert not r_a2_07_errors, (
        f"Normal source tidak boleh menghasilkan R-A2-07 error"
    )


# ─────────────────────────────────────────────
# R-A2-08 — INTEGRATION: MIXED SOURCE WARNING
# ─────────────────────────────────────────────

def test_r_a2_08_mixed_source_generates_warning():
    """
    Integration test: elemen dengan mixed source (content + non-content)
    yang lolos validate_element harus menghasilkan warning MIXED_SOURCE_NON_CONTENT
    di level run(), bukan failure.

    Memverifikasi bahwa:
    1. registry.failures kosong (tidak fatal)
    2. registry.warnings mengandung MIXED_SOURCE_NON_CONTENT
    3. warning menyebut element_id dan label yang tepat
    """
    mixed_element = {
        "label": "mixed_source_element",
        "element_type": "program_structure",
        "element_type_note": None,
        "summary": "Elemen dengan source campuran content dan non-content.",
        "source_block_refs": ["bab_7.B001", "bab_7.B003"],
        "source_block_fingerprints": [
            "bab_7__paragraph_lead__implementasi_program",
            "bab_7__divider__structural_2",
        ],
        "source_block_types": ["paragraph_lead", "divider"],
        "evidence_status": "inferred",
        "use_affordances": ["implementation_reference"],
        "guardrail_notes": ["Perlu diverifikasi lebih lanjut."],
    }

    registry, _ = run_with_mock_llm([mixed_element], mat=4, reu=4)

    # Tidak boleh fatal
    assert not registry.get("failures"), (
        f"Mixed source tidak boleh menghasilkan failures: {registry.get('failures')}"
    )

    # Harus ada warning MIXED_SOURCE_NON_CONTENT
    warnings = registry.get("warnings", [])
    mixed_warnings = [w for w in warnings if w.get("code") == "MIXED_SOURCE_NON_CONTENT"]
    assert mixed_warnings, (
        "Harus ada warning MIXED_SOURCE_NON_CONTENT untuk mixed source element"
    )

    # Warning harus menyebut element yang tepat
    w = mixed_warnings[0]
    assert "SUB-001" in w["message"], "Warning harus menyebut element_id"
    assert "mixed_source_element" in w["message"], "Warning harus menyebut label"
    assert "divider" in w["message"], "Warning harus menyebut non-content type yang ditemukan"
