"""
Tests untuk A2 Substance Extractor
Covers: validation rules R-A2-01..R-A2-14, assembly logic,
        priority formula, provenance checks, dry run.
LLM calls di-mock — tidak butuh API key.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import json
from unittest.mock import patch, MagicMock

from pipeline.a2_substance_extractor import (
    validate_affordances,
    validate_element,
    validate_provenance_against_packet,
    assemble_registry,
    build_pass1_prompt,
    build_pass2_prompt,
    parse_json_response,
    CORE_AFFORDANCES,
    VALID_ELEMENT_TYPES,
    VALID_EVIDENCE_STATUSES,
    run,
)


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

def make_pcb(**overrides):
    base = {
        "document_id": "test_doc_v1",
        "program_code": "TST",
        "substance_mode": "confirmed",
        "substance_chapter_id": "bab_7",
        "substance_basis": "manual_audit",
        "non_content_block_types": ["divider", "divider_thick", "spacer"],
    }
    base.update(overrides)
    return base


def make_block(ref, fp, btype, text=None):
    b = {"block_ref": ref, "block_fingerprint": fp, "original_index": 0, "type": btype}
    if text:
        b["text"] = text
    return b


def make_bip(chapters=None):
    if chapters is None:
        chapters = [{
            "chapter_id": "bab_7",
            "chapter_type": "implementation_core",
            "blocks": [
                make_block("bab_7.B001", "bab_7__paragraph_lead__implementasi", "paragraph_lead",
                           "Implementasi program dilakukan secara bertahap."),
                make_block("bab_7.B002", "bab_7__callout_info__sroi_blended", "callout_info",
                           "SROI blended program adalah 1:1.14."),
                make_block("bab_7.B003", "bab_7__divider__structural_2", "divider"),
                make_block("bab_7.B004", "bab_7__paragraph__dari_total_investasi", "paragraph",
                           "Dari total investasi Rp 1.355.826.539 dihasilkan nilai sosial positif."),
            ]
        }]
    return {
        "activity": "block_id_injector",
        "document_id": "test_doc_v1",
        "program_code": "TST",
        "generated_at": "2025-01-01T00:00:00+00:00",
        "chapters": chapters,
        "summary": {},
        "warnings": [],
        "failures": [],
    }


def make_valid_element(**overrides):
    base = {
        "element_id": "SUB-001",
        "label": "sroi_result",
        "element_type": "evaluative_metric",
        "element_type_note": None,
        "summary": "SROI blended program adalah 1:1.14.",
        "source_block_refs": ["bab_7.B002"],
        "source_block_fingerprints": ["bab_7__callout_info__sroi_blended"],
        "source_block_types": ["callout_info"],
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


def make_input_data(**pcb_overrides):
    return {
        "pipeline_control_block": make_pcb(**pcb_overrides),
        "block_identity_packet":  make_bip(),
    }


# ─────────────────────────────────────────────
# VALIDATE_AFFORDANCES
# ─────────────────────────────────────────────

class TestValidateAffordances:
    def test_core_affordances_valid(self):
        for aff in list(CORE_AFFORDANCES)[:5]:
            assert validate_affordances([aff]) == []

    def test_extension_x_valid(self):
        assert validate_affordances(["x_transition_pathway"]) == []
        assert validate_affordances(["x_employability_signal"]) == []

    def test_invalid_affordance(self):
        errors = validate_affordances(["invalid_affordance"])
        assert errors

    def test_mixed_valid_and_invalid(self):
        errors = validate_affordances(["opening_mandate", "bad_one"])
        assert len(errors) == 1

    def test_extension_without_x_prefix_invalid(self):
        errors = validate_affordances(["transition_pathway"])
        assert errors

    def test_empty_list_valid(self):
        assert validate_affordances([]) == []


# ─────────────────────────────────────────────
# VALIDATE_ELEMENT
# ─────────────────────────────────────────────

class TestValidateElement:
    def test_valid_element(self):
        el = make_valid_element()
        assert validate_element(el, 0, {"divider", "spacer"}) == []

    def test_missing_required_field(self):
        el = make_valid_element()
        del el["evidence_status"]
        errors = validate_element(el, 0, {"divider", "spacer"})
        assert any("evidence_status" in e for e in errors)

    def test_invalid_element_type(self):
        el = make_valid_element(element_type="invalid_type")
        errors = validate_element(el, 0, {"divider", "spacer"})
        assert any("element_type" in e for e in errors)

    def test_other_type_without_note(self):
        el = make_valid_element(element_type="other", element_type_note=None)
        errors = validate_element(el, 0, {"divider", "spacer"})
        assert any("element_type_note" in e for e in errors)

    def test_other_type_with_note_valid(self):
        el = make_valid_element(element_type="other", element_type_note="custom type")
        errors = validate_element(el, 0, {"divider", "spacer"})
        assert not errors

    def test_non_other_type_with_note_invalid(self):
        el = make_valid_element(element_type="evaluative_metric", element_type_note="should be null")
        errors = validate_element(el, 0, {"divider", "spacer"})
        assert any("element_type_note" in e for e in errors)

    def test_invalid_evidence_status(self):
        el = make_valid_element(evidence_status="uncertain")
        errors = validate_element(el, 0, {"divider", "spacer"})
        assert any("evidence_status" in e for e in errors)

    def test_provenance_length_mismatch(self):
        el = make_valid_element(
            source_block_refs=["bab_7.B001", "bab_7.B002"],
            source_block_fingerprints=["fp1"],
            source_block_types=["paragraph", "callout_info"],
        )
        errors = validate_element(el, 0, {"divider", "spacer"})
        assert any("panjang sama" in e for e in errors)

    def test_empty_source_arrays(self):
        el = make_valid_element(
            source_block_refs=[],
            source_block_fingerprints=[],
            source_block_types=[],
        )
        errors = validate_element(el, 0, {"divider", "spacer"})
        assert any("kosong" in e for e in errors)

    def test_empty_use_affordances(self):
        el = make_valid_element(use_affordances=[])
        errors = validate_element(el, 0, {"divider", "spacer"})
        assert any("use_affordances" in e for e in errors)

    def test_materiality_score_out_of_range(self):
        el = make_valid_element(materiality_score=6)
        errors = validate_element(el, 0, {"divider", "spacer"})
        assert any("materiality_score" in e for e in errors)

    def test_reusability_score_out_of_range(self):
        el = make_valid_element(reusability_score=0)
        errors = validate_element(el, 0, {"divider", "spacer"})
        assert any("reusability_score" in e for e in errors)

    def test_priority_inconsistent_high(self):
        el = make_valid_element(materiality_score=5, reusability_score=4, priority="medium")
        errors = validate_element(el, 0, {"divider", "spacer"})
        assert any("priority" in e for e in errors)

    def test_priority_inconsistent_low(self):
        el = make_valid_element(materiality_score=2, reusability_score=2, priority="high")
        errors = validate_element(el, 0, {"divider", "spacer"})
        assert any("priority" in e for e in errors)

    def test_priority_formula_boundaries(self):
        assert validate_element(make_valid_element(materiality_score=4, reusability_score=4, priority="high"), 0, {"divider"}) == []
        assert validate_element(make_valid_element(materiality_score=4, reusability_score=3, priority="medium"), 0, {"divider"}) == []
        assert validate_element(make_valid_element(materiality_score=3, reusability_score=2, priority="medium"), 0, {"divider"}) == []
        assert validate_element(make_valid_element(materiality_score=2, reusability_score=2, priority="low"), 0, {"divider"}) == []

    # ── R-A2-07 tests ─────────────────────────

    def test_r_a2_07_all_non_content_source_rejected(self):
        """R-A2-07: semua source non-content → fatal"""
        el = make_valid_element(
            source_block_refs=["bab_7.B003"],
            source_block_fingerprints=["bab_7__divider__structural_2"],
            source_block_types=["divider"],
        )
        errors = validate_element(el, 0, {"divider", "spacer", "divider_thick"})
        assert any("R-A2-07" in e for e in errors)

    def test_r_a2_07_multiple_non_content_rejected(self):
        """Dua non-content blocks sebagai satu-satunya source → fatal"""
        el = make_valid_element(
            source_block_refs=["bab_7.B003", "bab_7.B005"],
            source_block_fingerprints=["bab_7__divider__structural_2", "bab_7__spacer__structural_4"],
            source_block_types=["divider", "spacer"],
        )
        errors = validate_element(el, 0, {"divider", "spacer", "divider_thick"})
        assert any("R-A2-07" in e for e in errors)

    def test_r_a2_07_partial_non_content_allowed(self):
        """R-A2-08: mixed source (ada content + non-content) → tidak fatal di validate_element"""
        el = make_valid_element(
            source_block_refs=["bab_7.B001", "bab_7.B003"],
            source_block_fingerprints=[
                "bab_7__paragraph_lead__implementasi",
                "bab_7__divider__structural_2",
            ],
            source_block_types=["paragraph_lead", "divider"],
        )
        errors = validate_element(el, 0, {"divider", "spacer", "divider_thick"})
        assert not any("R-A2-07" in e for e in errors)

    def test_r_a2_07_normal_source_unaffected(self):
        """Source normal (tanpa non-content) → tidak ada error R-A2-07"""
        el = make_valid_element(
            source_block_types=["paragraph", "table_borderless"],
            source_block_refs=["bab_7.B001", "bab_7.B004"],
            source_block_fingerprints=[
                "bab_7__paragraph_lead__implementasi",
                "bab_7__paragraph__dari_total_investasi",
            ],
        )
        errors = validate_element(el, 0, {"divider", "spacer", "divider_thick"})
        assert not any("R-A2-07" in e for e in errors)


# ─────────────────────────────────────────────
# VALIDATE_PROVENANCE_AGAINST_PACKET
# ─────────────────────────────────────────────

class TestValidateProvenance:
    def test_valid_provenance(self):
        bip = make_bip()
        elements = [make_valid_element(
            source_block_refs=["bab_7.B002"],
            source_block_fingerprints=["bab_7__callout_info__sroi_blended"],
        )]
        assert validate_provenance_against_packet(elements, bip) == []

    def test_invalid_block_ref(self):
        bip = make_bip()
        elements = [make_valid_element(
            source_block_refs=["bab_7.B999"],  # tidak ada
            source_block_fingerprints=["bab_7__callout_info__sroi_blended"],
        )]
        errors = validate_provenance_against_packet(elements, bip)
        assert any("B999" in e for e in errors)

    def test_invalid_fingerprint(self):
        bip = make_bip()
        elements = [make_valid_element(
            source_block_refs=["bab_7.B002"],
            source_block_fingerprints=["bab_7__nonexistent__fp"],
        )]
        errors = validate_provenance_against_packet(elements, bip)
        assert any("nonexistent" in e for e in errors)

    def test_multi_source_valid(self):
        bip = make_bip()
        elements = [make_valid_element(
            source_block_refs=["bab_7.B001", "bab_7.B004"],
            source_block_fingerprints=[
                "bab_7__paragraph_lead__implementasi",
                "bab_7__paragraph__dari_total_investasi",
            ],
            source_block_types=["paragraph_lead", "paragraph"],
        )]
        assert validate_provenance_against_packet(elements, bip) == []


# ─────────────────────────────────────────────
# ASSEMBLE_REGISTRY
# ─────────────────────────────────────────────

class TestAssembleRegistry:
    def _make_pass1(self, n=2):
        return [
            {
                "label": f"element_{i}",
                "element_type": "evaluative_metric",
                "element_type_note": None,
                "summary": f"Summary elemen {i}.",
                "source_block_refs": ["bab_7.B001"],
                "source_block_fingerprints": ["bab_7__paragraph_lead__implementasi"],
                "source_block_types": ["paragraph_lead"],
                "evidence_status": "final",
                "use_affordances": ["opening_mandate"],
                "guardrail_notes": ["Guardrail note."],
            }
            for i in range(n)
        ]

    def _make_pass2(self, labels, scores=None):
        scored = []
        for i, label in enumerate(labels):
            s = (scores or {}).get(label, {"materiality_score": 4, "reusability_score": 4})
            scored.append({"label": label, **s})
        return {
            "scored_elements": scored,
            "global_guardrails": [
                {"guardrail_id": "SG-001", "scope": "document", "applies_to": "all",
                 "rule": "Global rule.", "severity": "medium"}
            ],
            "element_guardrails": [
                {"guardrail_id": "SG-E-001", "scope": "element",
                 "applies_to": "element_0", "rule": "Specific rule.", "severity": "low"}
            ],
        }

    def test_element_ids_assigned(self):
        p1 = self._make_pass1(3)
        p2 = self._make_pass2([e["label"] for e in p1])
        registry, _ = assemble_registry(p1, p2, "doc", "TST", "bab_7", "confirmed", "manual_audit")
        ids = [el["element_id"] for el in registry["elements"]]
        assert ids == ["SUB-001", "SUB-002", "SUB-003"]

    def test_priority_computed_from_scores(self):
        p1 = self._make_pass1(1)
        p2 = self._make_pass2(["element_0"], {"element_0": {"materiality_score": 5, "reusability_score": 4}})
        registry, _ = assemble_registry(p1, p2, "doc", "TST", "bab_7", "confirmed", "manual_audit")
        el = registry["elements"][0]
        assert el["materiality_score"] == 5
        assert el["reusability_score"] == 4
        assert el["priority"] == "high"

    def test_priority_medium(self):
        p1 = self._make_pass1(1)
        p2 = self._make_pass2(["element_0"], {"element_0": {"materiality_score": 3, "reusability_score": 3}})
        registry, _ = assemble_registry(p1, p2, "doc", "TST", "bab_7", "confirmed", "manual_audit")
        assert registry["elements"][0]["priority"] == "medium"

    def test_priority_low(self):
        p1 = self._make_pass1(1)
        p2 = self._make_pass2(["element_0"], {"element_0": {"materiality_score": 2, "reusability_score": 2}})
        registry, _ = assemble_registry(p1, p2, "doc", "TST", "bab_7", "confirmed", "manual_audit")
        assert registry["elements"][0]["priority"] == "low"

    def test_guardrail_ids_renumbered(self):
        p1 = self._make_pass1(1)
        p2 = self._make_pass2(["element_0"])
        _, guardrails = assemble_registry(p1, p2, "doc", "TST", "bab_7", "confirmed", "manual_audit")
        assert guardrails["global_guardrails"][0]["guardrail_id"] == "SG-001"
        assert guardrails["element_guardrails"][0]["guardrail_id"] == "SG-E-001"

    def test_element_guardrail_applies_to_resolved(self):
        """applies_to dari label harus diresolve ke element_id."""
        p1 = self._make_pass1(1)
        p2 = self._make_pass2(["element_0"])
        _, guardrails = assemble_registry(p1, p2, "doc", "TST", "bab_7", "confirmed", "manual_audit")
        eg = guardrails["element_guardrails"][0]
        assert eg["applies_to"] == "SUB-001"

    def test_summary_counts(self):
        p1 = self._make_pass1(3)
        p2 = self._make_pass2([e["label"] for e in p1])
        registry, _ = assemble_registry(p1, p2, "doc", "TST", "bab_7", "confirmed", "manual_audit")
        assert registry["summary"]["total_elements"] == 3

    def test_status_active(self):
        p1 = self._make_pass1(2)
        p2 = self._make_pass2([e["label"] for e in p1])
        registry, _ = assemble_registry(p1, p2, "doc", "TST", "bab_7", "confirmed", "manual_audit")
        for el in registry["elements"]:
            assert el["status"] == "active"

    def test_empty_pass1(self):
        registry, guardrails = assemble_registry(
            [], {}, "doc", "TST", "bab_7", "confirmed", "manual_audit"
        )
        assert registry["elements"] == []
        assert registry["summary"]["total_elements"] == 0

    def test_metadata_fields(self):
        p1 = self._make_pass1(1)
        p2 = self._make_pass2(["element_0"])
        registry, guardrails = assemble_registry(
            p1, p2, "my_doc", "ESD", "bab_7", "confirmed", "manual_audit"
        )
        assert registry["activity"] == "substance_extractor"
        assert registry["document_id"] == "my_doc"
        assert registry["program_code"] == "ESD"
        assert guardrails["substance_chapter_id"] == "bab_7"


# ─────────────────────────────────────────────
# PARSE_JSON_RESPONSE
# ─────────────────────────────────────────────

class TestParseJsonResponse:
    def test_valid_json(self):
        data, err = parse_json_response('{"elements": []}', "test")
        assert err is None
        assert data == {"elements": []}

    def test_invalid_json(self):
        data, err = parse_json_response("not json", "test")
        assert err is not None
        assert data is None

    def test_array_json(self):
        data, err = parse_json_response('[{"a": 1}]', "test")
        assert err is None
        assert data == [{"a": 1}]


# ─────────────────────────────────────────────
# BUILD PROMPTS
# ─────────────────────────────────────────────

class TestBuildPrompts:
    def test_pass1_prompt_contains_chapter_id(self):
        blocks = [make_block("bab_7.B001", "fp1", "paragraph", "teks")]
        prompt = build_pass1_prompt("bab_7", blocks, {"divider"})
        assert "bab_7" in prompt

    def test_pass1_prompt_marks_non_content_blocks(self):
        blocks = [
            make_block("bab_7.B001", "fp1", "paragraph", "teks"),
            make_block("bab_7.B002", "fp2", "divider"),
        ]
        prompt = build_pass1_prompt("bab_7", blocks, {"divider"})
        assert "NON-CONTENT" in prompt

    def test_pass1_prompt_includes_block_ref(self):
        blocks = [make_block("bab_7.B001", "fp1", "paragraph", "teks")]
        prompt = build_pass1_prompt("bab_7", blocks, set())
        assert "bab_7.B001" in prompt

    def test_pass2_prompt_contains_elements(self):
        elements = [{"label": "sroi_result", "summary": "SROI 1:1.14"}]
        prompt = build_pass2_prompt(elements, "doc context")
        assert "sroi_result" in prompt
        assert "doc context" in prompt


# ─────────────────────────────────────────────
# RUN — DRY RUN (no LLM)
# ─────────────────────────────────────────────

class TestRunDryRun:
    def test_dry_run_returns_empty_registry(self):
        input_data = make_input_data()
        registry, guardrails = run(input_data, dry_run=True)
        assert registry["activity"] == "substance_extractor"
        assert registry["elements"] == []
        assert registry["document_id"] == "test_doc_v1"

    def test_dry_run_summary_complete(self):
        """Dry run summary harus punya warning_count dan failure_count."""
        input_data = make_input_data()
        registry, _ = run(input_data, dry_run=True)
        summary = registry["summary"]
        assert "warning_count" in summary
        assert "failure_count" in summary
        assert summary["warning_count"] == 0
        assert summary["failure_count"] == 0

    def test_dry_run_returns_guardrails_skeleton(self):
        input_data = make_input_data()
        _, guardrails = run(input_data, dry_run=True)
        assert "global_guardrails" in guardrails
        assert "element_guardrails" in guardrails

    def test_dry_run_resolves_source_chapter(self):
        input_data = make_input_data()
        registry, _ = run(input_data, dry_run=True)
        assert registry["substance_chapter_id"] == "bab_7"

    def test_dry_run_discovery_mode_picks_first_chapter(self):
        input_data = {
            "pipeline_control_block": make_pcb(
                substance_mode="discovery",
                substance_chapter_id=None,
            ),
            "block_identity_packet": make_bip(),
        }
        registry, _ = run(input_data, dry_run=True)
        assert registry["substance_chapter_id"] == "bab_7"

    def test_dry_run_raises_if_chapter_not_found(self):
        input_data = make_input_data(substance_chapter_id="bab_99")
        with pytest.raises(ValueError, match="bab_99"):
            run(input_data, dry_run=True)

    def test_dry_run_raises_if_confirmed_without_chapter_id(self):
        input_data = {
            "pipeline_control_block": make_pcb(
                substance_mode="confirmed",
                substance_chapter_id=None,
            ),
            "block_identity_packet": make_bip(),
        }
        with pytest.raises(ValueError, match="substance_chapter_id"):
            run(input_data, dry_run=True)


# ─────────────────────────────────────────────
# RUN — MOCKED LLM
# ─────────────────────────────────────────────

def make_mock_pass1_response():
    elements = [
        {
            "label": "sroi_blended_result",
            "element_type": "evaluative_metric",
            "element_type_note": None,
            "summary": "SROI blended program adalah 1:1.14 untuk periode evaluasi.",
            "source_block_refs": ["bab_7.B002"],
            "source_block_fingerprints": ["bab_7__callout_info__sroi_blended"],
            "source_block_types": ["callout_info"],
            "evidence_status": "final",
            "use_affordances": ["opening_mandate", "closing_summary"],
            "guardrail_notes": ["Jangan disebutkan terlalu dini sebagai angka dominan."],
        },
        {
            "label": "investment_structure",
            "element_type": "investment_structure",
            "element_type_note": None,
            "summary": "Total investasi program adalah Rp 1.355.826.539.",
            "source_block_refs": ["bab_7.B004"],
            "source_block_fingerprints": ["bab_7__paragraph__dari_total_investasi"],
            "source_block_types": ["paragraph"],
            "evidence_status": "final",
            "use_affordances": ["implementation_reference", "monetization_reference"],
            "guardrail_notes": ["Selalu sertakan satuan mata uang."],
        },
    ]
    return json.dumps({"elements": elements})


def make_mock_pass2_response():
    return json.dumps({
        "scored_elements": [
            {"label": "sroi_blended_result", "materiality_score": 5, "reusability_score": 5},
            {"label": "investment_structure", "materiality_score": 4, "reusability_score": 3},
        ],
        "global_guardrails": [
            {
                "guardrail_id": "SG-001",
                "scope": "document",
                "applies_to": "all",
                "rule": "Semua klaim numerik harus disertai konteks periode.",
                "severity": "medium",
            }
        ],
        "element_guardrails": [
            {
                "guardrail_id": "SG-E-001",
                "scope": "element",
                "applies_to": "sroi_blended_result",
                "rule": "Jangan pakai sebagai angka pembuka bab.",
                "severity": "medium",
            }
        ],
    })


class TestRunMockedLLM:
    def _run_with_mock(self, input_data=None):
        if input_data is None:
            input_data = make_input_data()

        mock_choice = MagicMock()
        mock_choice.message.content = make_mock_pass1_response()

        mock_choice2 = MagicMock()
        mock_choice2.message.content = make_mock_pass2_response()

        call_count = [0]
        def side_effect(**kwargs):
            call_count[0] += 1
            resp = MagicMock()
            if call_count[0] == 1:
                resp.choices = [mock_choice]
            else:
                resp.choices = [mock_choice2]
            return resp

        with patch("pipeline.a2_substance_extractor.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = side_effect
            MockOpenAI.return_value = mock_client
            registry, guardrails = run(input_data, api_key="test-key")

        return registry, guardrails

    def test_two_llm_calls_made(self):
        input_data = make_input_data()
        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            resp = MagicMock()
            choice = MagicMock()
            if call_count[0] == 1:
                choice.message.content = make_mock_pass1_response()
            else:
                choice.message.content = make_mock_pass2_response()
            resp.choices = [choice]
            return resp

        with patch("pipeline.a2_substance_extractor.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = side_effect
            MockOpenAI.return_value = mock_client
            run(input_data, api_key="test-key")

        assert call_count[0] == 2

    def test_elements_assembled_correctly(self):
        registry, _ = self._run_with_mock()
        assert len(registry["elements"]) == 2
        labels = [el["label"] for el in registry["elements"]]
        assert "sroi_blended_result" in labels
        assert "investment_structure" in labels

    def test_element_ids_sequential(self):
        registry, _ = self._run_with_mock()
        ids = [el["element_id"] for el in registry["elements"]]
        assert ids == ["SUB-001", "SUB-002"]

    def test_priority_computed_correctly(self):
        registry, _ = self._run_with_mock()
        el = next(e for e in registry["elements"] if e["label"] == "sroi_blended_result")
        assert el["materiality_score"] == 5
        assert el["reusability_score"] == 5
        assert el["priority"] == "high"  # 5+5=10 >= 8

        el2 = next(e for e in registry["elements"] if e["label"] == "investment_structure")
        assert el2["priority"] == "medium"  # 4+3=7

    def test_no_failures_on_valid_output(self):
        registry, _ = self._run_with_mock()
        assert not registry.get("failures")

    def test_guardrails_structure(self):
        _, guardrails = self._run_with_mock()
        assert len(guardrails["global_guardrails"]) == 1
        assert len(guardrails["element_guardrails"]) == 1
        assert guardrails["global_guardrails"][0]["guardrail_id"] == "SG-001"
        assert guardrails["element_guardrails"][0]["guardrail_id"] == "SG-E-001"

    def test_element_guardrail_applies_to_resolved(self):
        _, guardrails = self._run_with_mock()
        eg = guardrails["element_guardrails"][0]
        # "sroi_blended_result" harus diresolve ke "SUB-001"
        assert eg["applies_to"] == "SUB-001"

    def test_r_a2_03_provenance_validated(self):
        """R-A2-03: provenance divalidasi terhadap block_identity_packet."""
        # Modifikasi mock agar return fingerprint yang tidak ada di BIP
        bad_pass1 = json.dumps({"elements": [{
            "label": "bad_element",
            "element_type": "evaluative_metric",
            "element_type_note": None,
            "summary": "Bad element.",
            "source_block_refs": ["bab_7.B999"],  # tidak ada di BIP
            "source_block_fingerprints": ["bab_7__nonexistent__fp"],
            "source_block_types": ["paragraph"],
            "evidence_status": "final",
            "use_affordances": ["opening_mandate"],
            "guardrail_notes": ["note"],
        }]})

        def side_effect(**kwargs):
            call_count[0] += 1
            resp = MagicMock()
            choice = MagicMock()
            if call_count[0] == 1:
                choice.message.content = bad_pass1
            else:
                choice.message.content = json.dumps({
                    "scored_elements": [{"label": "bad_element", "materiality_score": 3, "reusability_score": 3}],
                    "global_guardrails": [],
                    "element_guardrails": [],
                })
            resp.choices = [choice]
            return resp

        call_count = [0]
        input_data = make_input_data()

        with patch("pipeline.a2_substance_extractor.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = side_effect
            MockOpenAI.return_value = mock_client
            registry, _ = run(input_data, api_key="test-key")

        assert registry.get("failures")
        assert any("B999" in f["message"] for f in registry["failures"])

    def test_r_a2_01_source_from_control_block(self):
        """R-A2-01: source chapter dari pipeline_control_block."""
        # Ubah substance_chapter_id → bab_1, buat BIP dengan bab_1
        bip_with_bab1 = make_bip([{
            "chapter_id": "bab_1",
            "chapter_type": "opening",
            "blocks": [make_block("bab_1.B001", "bab_1__para__teks", "paragraph", "teks")]
        }])
        input_data = {
            "pipeline_control_block": make_pcb(substance_chapter_id="bab_1"),
            "block_identity_packet": bip_with_bab1,
        }
        registry, _ = run(input_data, dry_run=True)
        assert registry["substance_chapter_id"] == "bab_1"

    def test_r_a2_10_evidence_status_preserved(self):
        """R-A2-10: evidence_status dari LLM dipertahankan as-is."""
        registry, _ = self._run_with_mock()
        for el in registry["elements"]:
            assert el["evidence_status"] in VALID_EVIDENCE_STATUSES

    def test_r_a2_07_non_content_only_source_rejected_end_to_end(self):
        """R-A2-07 end-to-end: LLM return elemen dengan source divider saja → failures"""
        bad_pass1 = json.dumps({"elements": [{
            "label": "divider_element",
            "element_type": "evaluative_metric",
            "element_type_note": None,
            "summary": "Elemen yang sumbernya hanya divider.",
            "source_block_refs": ["bab_7.B003"],
            "source_block_fingerprints": ["bab_7__divider__structural_2"],
            "source_block_types": ["divider"],   # non-content only → harus fatal
            "evidence_status": "final",
            "use_affordances": ["opening_mandate"],
            "guardrail_notes": ["note"],
        }]})

        call_count = [0]
        def side_effect(**kwargs):
            call_count[0] += 1
            resp = MagicMock()
            choice = MagicMock()
            if call_count[0] == 1:
                choice.message.content = bad_pass1
            else:
                choice.message.content = json.dumps({
                    "scored_elements": [{"label": "divider_element", "materiality_score": 3, "reusability_score": 3}],
                    "global_guardrails": [],
                    "element_guardrails": [],
                })
            resp.choices = [choice]
            return resp

        input_data = make_input_data()
        with patch("pipeline.a2_substance_extractor.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = side_effect
            MockOpenAI.return_value = mock_client
            registry, _ = run(input_data, api_key="test-key")

        assert registry.get("failures"), "Harus ada failures untuk non-content-only source"
        assert any("R-A2-07" in f["message"] for f in registry["failures"])

    def test_r_a2_08_mixed_source_produces_warning_not_failure(self):
        """R-A2-08: mixed source (paragraph + divider) → warning, bukan failure"""
        mixed_pass1 = json.dumps({"elements": [{
            "label": "mixed_element",
            "element_type": "evaluative_metric",
            "element_type_note": None,
            "summary": "Elemen dengan source campuran.",
            "source_block_refs": ["bab_7.B001", "bab_7.B003"],
            "source_block_fingerprints": [
                "bab_7__paragraph_lead__implementasi",
                "bab_7__divider__structural_2",
            ],
            "source_block_types": ["paragraph_lead", "divider"],
            "evidence_status": "final",
            "use_affordances": ["opening_mandate"],
            "guardrail_notes": ["note"],
        }]})

        call_count = [0]
        def side_effect(**kwargs):
            call_count[0] += 1
            resp = MagicMock()
            choice = MagicMock()
            if call_count[0] == 1:
                choice.message.content = mixed_pass1
            else:
                choice.message.content = json.dumps({
                    "scored_elements": [{"label": "mixed_element", "materiality_score": 4, "reusability_score": 4}],
                    "global_guardrails": [],
                    "element_guardrails": [],
                })
            resp.choices = [choice]
            return resp

        input_data = make_input_data()
        with patch("pipeline.a2_substance_extractor.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = side_effect
            MockOpenAI.return_value = mock_client
            registry, _ = run(input_data, api_key="test-key")

        assert not registry.get("failures"), "Mixed source tidak boleh menghasilkan failures"
        warnings = registry.get("warnings", [])
        assert any(w["code"] == "MIXED_SOURCE_NON_CONTENT" for w in warnings)

    def test_summary_contains_warning_count(self):
        """warning_count harus ada di summary."""
        registry, _ = self._run_with_mock()
        assert "warning_count" in registry["summary"]

    def test_r_a2_11_affordances_validated(self):
        """R-A2-11: affordances divalidasi."""
        registry, _ = self._run_with_mock()
        for el in registry["elements"]:
            for aff in el["use_affordances"]:
                is_core = aff in CORE_AFFORDANCES
                is_ext  = bool(__import__("re").match(r"^x_[a-z0-9_]+$", aff))
                assert is_core or is_ext, f"Affordance tidak valid: {aff}"
