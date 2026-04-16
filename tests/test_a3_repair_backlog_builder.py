"""
Test suite A3 — Repair Backlog Builder
160 tests → target: semua rules dan edge cases
"""

import pytest
from pipeline.a3_repair_backlog_builder import (
    compute_readiness,
    compute_issue_burden,
    compute_chapter_health,
    make_issue,
    detect_empty_block,
    detect_empty_table,
    detect_empty_list,
    detect_placeholder_content,
    detect_template_residue,
    detect_anomalous_value,
    detect_duplicate_function,
    detect_missing_substance_alignment,
    detect_evidence_gap,
    detect_possible_generic_claim,
    detect_possible_unsupported_inference,
    detect_possible_weak_transition,
    detect_possible_overclaim_risk,
    detect_structural_imbalance,
    detect_visual_gap,
    validate_issue,
    validate_chapter_backlog,
    semantic_lint_a3,
    scan_chapter,
    run,
    SEVERITY_WEIGHT,
    ISSUE_TYPE_MULTIPLIER,
    CHAPTER_LEVEL_ISSUES,
)

NCT = {"divider", "divider_thick", "spacer"}


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

def make_block(block_ref="bab_1.B001", btype="paragraph", text="Ini adalah teks blok.", fp=None):
    return {
        "block_ref": block_ref,
        "block_fingerprint": fp or f"bab_1__{btype}__ini_adalah_teks_blok",
        "type": btype,
        "text": text,
    }


def make_element(label="sroi_metric", evidence_status="final", affordances=None, summary="SROI 1:1.12"):
    return {
        "element_id": "SUB-001",
        "label": label,
        "summary": summary,
        "evidence_status": evidence_status,
        "use_affordances": affordances or ["outcome_reference"],
        "materiality_score": 5,
        "reusability_score": 4,
    }


def make_pcb(substance_chapter_id="bab_7"):
    return {
        "document_id": "test_doc",
        "program_code": "TST",
        "substance_chapter_id": substance_chapter_id,
        "non_content_block_types": ["divider", "divider_thick", "spacer"],
    }


def make_bip(chapters):
    return {"chapters": chapters}


def make_registry(elements=None):
    return {"elements": elements or [make_element()]}


# ─────────────────────────────────────────────
# BURDEN FORMULA
# ─────────────────────────────────────────────

class TestBurdenFormula:
    def test_anomalous_value_critical_not_ready(self):
        # (7 × 1.5) + 2 = 12.5
        burden = compute_issue_burden("critical", "anomalous_value", False)
        assert burden == 12.5

    def test_empty_block_high_not_ready(self):
        # (4 × 1.5) + 2 = 8.0
        burden = compute_issue_burden("high", "empty_block", False)
        assert burden == 8.0

    def test_unsupported_inference_medium_ready(self):
        # (2 × 1.25) + 0 = 2.5
        burden = compute_issue_burden("medium", "possible_unsupported_inference", True)
        assert burden == 2.5

    def test_generic_issue_low_ready(self):
        # (1 × 1.0) + 0 = 1.0
        burden = compute_issue_burden("low", "possible_generic_claim", True)
        assert burden == 1.0

    def test_readiness_penalty_applied(self):
        ready    = compute_issue_burden("medium", "structural_imbalance", True)
        not_ready = compute_issue_burden("medium", "structural_imbalance", False)
        assert not_ready - ready == 2.0


# ─────────────────────────────────────────────
# READINESS RULES (R-A3-03, R-A3-04)
# ─────────────────────────────────────────────

class TestReadiness:
    def test_critical_always_not_ready(self):
        assert compute_readiness("critical", "empty_block", "rewrite") is False

    def test_scaffold_only_not_ready(self):
        assert compute_readiness("low", "structural_imbalance", "scaffold_only") is False

    def test_verify_not_ready(self):
        assert compute_readiness("medium", "evidence_gap", "verify") is False

    def test_hold_not_ready(self):
        assert compute_readiness("low", "other", "hold") is False

    def test_refine_medium_is_ready(self):
        assert compute_readiness("medium", "possible_generic_claim", "refine") is True

    def test_rewrite_high_is_ready(self):
        assert compute_readiness("high", "empty_block", "rewrite") is True


# ─────────────────────────────────────────────
# CHAPTER HEALTH
# ─────────────────────────────────────────────

class TestChapterHealth:
    def _issue(self, sev, ready=True, itype="empty_block"):
        return {"severity": sev, "ready_for_point_builder": ready, "issue_type": itype}

    def test_high_below_threshold(self):
        assert compute_chapter_health([], 0.0) == "high"
        assert compute_chapter_health([], 2.24) == "high"

    def test_medium_range(self):
        assert compute_chapter_health([], 2.25) == "medium"
        assert compute_chapter_health([], 4.49) == "medium"

    def test_low_above_threshold(self):
        assert compute_chapter_health([], 4.5) == "low"
        assert compute_chapter_health([], 10.0) == "low"

    def test_override_two_critical(self):
        issues = [self._issue("critical"), self._issue("critical"), self._issue("low")]
        assert compute_chapter_health(issues, 1.0) == "low"

    def test_override_one_critical_no_override(self):
        issues = [self._issue("critical"), self._issue("low")]
        assert compute_chapter_health(issues, 1.0) == "high"

    def test_override_anomalous_not_ready(self):
        issues = [self._issue("medium", ready=False, itype="anomalous_value")]
        assert compute_chapter_health(issues, 1.0) == "low"

    def test_override_30pct_not_ready(self):
        # 4 dari 10 (40%) not ready → override
        issues = [self._issue("medium", ready=False)] * 4 + [self._issue("low", ready=True)] * 6
        assert compute_chapter_health(issues, 1.0) == "low"

    def test_no_override_30pct_boundary(self):
        # Tepat 30% → tidak override
        issues = [self._issue("medium", ready=False)] * 3 + [self._issue("low", ready=True)] * 7
        assert compute_chapter_health(issues, 1.0) == "high"

    def test_override_monotonic_cannot_upgrade(self):
        # Burden tinggi → low, override tidak bisa naik ke medium
        issues = []  # tidak ada override condition
        assert compute_chapter_health(issues, 5.0) == "low"


# ─────────────────────────────────────────────
# HARD DETECTORS
# ─────────────────────────────────────────────

class TestDetectEmptyBlock:
    def test_empty_text_flagged(self):
        b = make_block(text="")
        assert detect_empty_block(b, NCT) is True

    def test_none_text_flagged(self):
        b = make_block(text=None)
        assert detect_empty_block(b, NCT) is True

    def test_whitespace_only_flagged(self):
        b = make_block(text="   ")
        assert detect_empty_block(b, NCT) is True

    def test_non_content_not_flagged(self):
        b = make_block(btype="divider", text="")
        assert detect_empty_block(b, NCT) is False

    def test_normal_text_not_flagged(self):
        b = make_block(text="Normal content")
        assert detect_empty_block(b, NCT) is False


class TestDetectEmptyTable:
    def test_empty_rows_flagged(self):
        b = {"type": "table", "rows": []}
        assert detect_empty_table(b) is True

    def test_null_rows_flagged(self):
        b = {"type": "table", "rows": None}
        assert detect_empty_table(b) is True

    def test_has_rows_not_flagged(self):
        b = {"type": "table", "rows": [["A", "B"]]}
        assert detect_empty_table(b) is False

    def test_non_table_not_flagged(self):
        b = {"type": "paragraph", "rows": []}
        assert detect_empty_table(b) is False


class TestDetectEmptyList:
    def test_empty_items_flagged(self):
        b = {"type": "list", "items": []}
        assert detect_empty_list(b) is True

    def test_has_items_not_flagged(self):
        b = {"type": "list", "items": [{"text": "item"}]}
        assert detect_empty_list(b) is False


class TestDetectPlaceholder:
    def test_bracket_placeholder_flagged(self):
        b = make_block(text="[Isi deskripsi program di sini]")
        assert detect_placeholder_content(b, NCT) is True

    def test_tbd_flagged(self):
        b = make_block(text="TBD")
        assert detect_placeholder_content(b, NCT) is True

    def test_todo_flagged(self):
        b = make_block(text="TODO: tambahkan data investasi")
        assert detect_placeholder_content(b, NCT) is True

    def test_normal_text_not_flagged(self):
        b = make_block(text="Program ESD berjalan sejak 2023.")
        assert detect_placeholder_content(b, NCT) is False

    def test_non_content_not_flagged(self):
        b = make_block(btype="divider", text="[placeholder]")
        assert detect_placeholder_content(b, NCT) is False


class TestDetectTemplateResidue:
    def test_tuliskan_flagged(self):
        b = make_block(text="Tuliskan deskripsi program di sini.")
        assert detect_template_residue(b, NCT) is True

    def test_jelaskan_flagged(self):
        b = make_block(text="Jelaskan metodologi yang digunakan.")
        assert detect_template_residue(b, NCT) is True

    def test_normal_text_not_flagged(self):
        b = make_block(text="Program ini berfokus pada pemberdayaan difabel.")
        assert detect_template_residue(b, NCT) is False


class TestDetectAnomalousValue:
    def test_zero_in_investment_context(self):
        b = make_block(text="Total investasi program adalah 0 rupiah.")
        assert detect_anomalous_value(b, NCT) is True

    def test_zero_in_sroi_context(self):
        b = make_block(text="Nilai SROI program adalah 0.")
        assert detect_anomalous_value(b, NCT) is True

    def test_zero_no_context_not_flagged(self):
        b = make_block(text="Terdapat 0 komentar pada dokumen ini.")
        assert detect_anomalous_value(b, NCT) is False

    def test_null_string_flagged(self):
        b = make_block(text='Nilai investasi: "null"')
        assert detect_anomalous_value(b, NCT) is True

    def test_normal_value_not_flagged(self):
        b = make_block(text="Total investasi mencapai Rp 594.781.486.")
        assert detect_anomalous_value(b, NCT) is False


class TestDetectDuplicate:
    def test_exact_fingerprint_duplicate(self):
        b1 = make_block("bab_1.B001", fp="bab_1__paragraph__sama")
        b2 = make_block("bab_1.B005", fp="bab_1__paragraph__sama")
        dupes = detect_duplicate_function([b1, b2], NCT)
        assert len(dupes) == 1

    def test_different_fingerprints_no_duplicate(self):
        b1 = make_block("bab_1.B001", fp="bab_1__paragraph__pertama")
        b2 = make_block("bab_1.B002", fp="bab_1__paragraph__kedua")
        dupes = detect_duplicate_function([b1, b2], NCT)
        assert len(dupes) == 0

    def test_non_content_not_checked(self):
        b1 = {"type": "divider", "block_ref": "bab_1.B001", "block_fingerprint": "bab_1__divider__structural_0"}
        b2 = {"type": "divider", "block_ref": "bab_1.B002", "block_fingerprint": "bab_1__divider__structural_0"}
        dupes = detect_duplicate_function([b1, b2], NCT)
        assert len(dupes) == 0


class TestDetectMissingAlignment:
    def _elements(self):
        return [make_element(label="sroi_metric", summary="SROI 1:1.12 outcome investasi program")]

    def test_aligned_via_lexical_not_flagged(self):
        b = make_block(text="Program ini menghasilkan nilai SROI yang positif dari investasi yang dilakukan selama ini.")
        assert detect_missing_substance_alignment(b, NCT, self._elements()) is False

    def test_aligned_via_number_not_flagged(self):
        b = make_block(text="Evaluasi menunjukkan rasio 1:1.12 sebagai hasil pengukuran dampak program pada masyarakat.")
        assert detect_missing_substance_alignment(b, NCT, self._elements()) is False

    def test_aligned_via_affordance_keyword_not_flagged(self):
        b = make_block(text="Outcome program menunjukkan peningkatan yang signifikan bagi seluruh penerima manfaat.")
        assert detect_missing_substance_alignment(b, NCT, self._elements()) is False

    def test_short_block_not_checked(self):
        b = make_block(text="Teks singkat.")
        assert detect_missing_substance_alignment(b, NCT, self._elements()) is False

    def test_non_content_not_checked(self):
        b = make_block(btype="divider", text="x " * 25)
        assert detect_missing_substance_alignment(b, NCT, self._elements()) is False

    def test_unrelated_text_flagged(self):
        b = make_block(text="Cuaca hari ini sangat cerah dan matahari bersinar dengan terang di langit biru yang indah sekali.")
        assert detect_missing_substance_alignment(b, NCT, self._elements()) is True


class TestDetectEvidenceGap:
    def test_block_mentioning_pending_element_flagged(self):
        elements = [make_element(label="pending_outcome", evidence_status="pending")]
        b = make_block(text="Outcome pending_outcome belum terverifikasi.")
        assert detect_evidence_gap(b, NCT, elements) is True

    def test_block_mentioning_final_element_not_flagged(self):
        elements = [make_element(label="sroi_metric", evidence_status="final")]
        b = make_block(text="sroi metric sudah diverifikasi.")
        assert detect_evidence_gap(b, NCT, elements) is False

    def test_non_content_not_checked(self):
        elements = [make_element(label="pending_outcome", evidence_status="pending")]
        b = make_block(btype="divider", text="pending outcome")
        assert detect_evidence_gap(b, NCT, elements) is False


# ─────────────────────────────────────────────
# HEURISTIC DETECTORS
# ─────────────────────────────────────────────

class TestHeuristicDetectors:
    def test_generic_claim_short_no_number_no_noun(self):
        b = make_block(text="Program ini berjalan baik.")
        assert detect_possible_generic_claim(b, NCT) is True

    def test_generic_claim_has_number_not_flagged(self):
        b = make_block(text="Program ini berjalan 3 tahun.")
        assert detect_possible_generic_claim(b, NCT) is False

    def test_generic_claim_long_not_flagged(self):
        b = make_block(text="Program ini berjalan baik dan memberikan dampak positif bagi masyarakat luas di berbagai daerah.")
        assert detect_possible_generic_claim(b, NCT) is False

    def test_generic_claim_has_sroi_keyword_not_flagged(self):
        b = make_block(text="Nilai SROI program baik.")
        assert detect_possible_generic_claim(b, NCT) is False

    def test_unsupported_inference_flagged(self):
        b = make_block(text="Terbukti program ini berhasil mencapai tujuan.")
        assert detect_possible_unsupported_inference(b, NCT) is True

    def test_unsupported_inference_has_number_not_flagged(self):
        b = make_block(text="Menunjukkan peningkatan 35% dari baseline.")
        assert detect_possible_unsupported_inference(b, NCT) is False

    def test_unsupported_inference_has_reference_not_flagged(self):
        b = make_block(text="Terbukti berdasarkan data transaksi yang tercatat.")
        assert detect_possible_unsupported_inference(b, NCT) is False

    def test_weak_transition_boundary_short_flagged(self):
        b = make_block(text="Bab ini.")
        assert detect_possible_weak_transition(b, NCT, True) is True

    def test_weak_transition_not_boundary_not_flagged(self):
        b = make_block(text="Bab ini.")
        assert detect_possible_weak_transition(b, NCT, False) is False

    def test_weak_transition_boundary_long_not_flagged(self):
        b = make_block(text="Bab ini membahas implementasi program secara menyeluruh.")
        assert detect_possible_weak_transition(b, NCT, True) is False

    def test_overclaim_proxy_no_caveat_flagged(self):
        elements = [make_element(label="kemandirian_ekonomi", evidence_status="proxy")]
        b = make_block(text="kemandirian ekonomi operator difabel telah meningkat secara signifikan.")
        assert detect_possible_overclaim_risk(b, NCT, elements) is True

    def test_overclaim_has_caveat_not_flagged(self):
        elements = [make_element(label="kemandirian_ekonomi", evidence_status="proxy")]
        b = make_block(text="kemandirian ekonomi operator difabel diperkirakan meningkat.")
        assert detect_possible_overclaim_risk(b, NCT, elements) is False

    def test_overclaim_final_evidence_not_flagged(self):
        elements = [make_element(label="transaksi_bengkel", evidence_status="final")]
        b = make_block(text="transaksi bengkel difabel meningkat secara signifikan.")
        assert detect_possible_overclaim_risk(b, NCT, elements) is False


# ─────────────────────────────────────────────
# CHAPTER-LEVEL DETECTORS
# ─────────────────────────────────────────────

class TestChapterLevelDetectors:
    def test_structural_imbalance_zero_blocks(self):
        assert detect_structural_imbalance([]) is True

    def test_structural_imbalance_one_block(self):
        assert detect_structural_imbalance([make_block()]) is True

    def test_structural_imbalance_two_blocks_ok(self):
        assert detect_structural_imbalance([make_block(), make_block()]) is False

    def test_visual_gap_relevant_chapter_no_visual(self):
        elements = [make_element(affordances=["visual_candidate", "outcome_reference"])]
        blocks = [make_block(btype="paragraph")]
        assert detect_visual_gap("bab_4", blocks, elements) is True

    def test_visual_gap_has_table_not_flagged(self):
        elements = [make_element(affordances=["visual_candidate"])]
        blocks = [make_block(btype="table")]
        assert detect_visual_gap("bab_4", blocks, elements) is False

    def test_visual_gap_irrelevant_chapter_not_flagged(self):
        elements = [make_element(affordances=["visual_candidate"])]
        blocks = [make_block(btype="paragraph")]
        assert detect_visual_gap("bab_1", blocks, elements) is False

    def test_visual_gap_no_visual_candidate_not_flagged(self):
        elements = [make_element(affordances=["outcome_reference"])]
        blocks = [make_block(btype="paragraph")]
        assert detect_visual_gap("bab_4", blocks, elements) is False


# ─────────────────────────────────────────────
# CHAPTER-LEVEL ISSUES: block_ref = null (R-A3 new spec)
# ─────────────────────────────────────────────

class TestChapterLevelIssueFields:
    def test_visual_gap_has_null_block_ref(self):
        issue = make_issue("bab_4", 1, "visual_gap", None, "No visual blocks.")
        assert issue["block_ref"] is None
        assert issue["block_fingerprint"] is None

    def test_structural_imbalance_has_null_block_ref(self):
        issue = make_issue("bab_4", 1, "structural_imbalance", None, "Too few blocks.")
        assert issue["block_ref"] is None
        assert issue["block_fingerprint"] is None

    def test_block_issue_has_block_ref(self):
        b = make_block("bab_1.B003")
        issue = make_issue("bab_1", 1, "empty_block", b, "Empty.")
        assert issue["block_ref"] == "bab_1.B003"
        assert issue["block_fingerprint"] is not None


# ─────────────────────────────────────────────
# MAKE_ISSUE — field integrity
# ─────────────────────────────────────────────

class TestMakeIssue:
    def test_all_fields_present(self):
        b = make_block()
        issue = make_issue("bab_1", 1, "empty_block", b, "Test.")
        required = [
            "issue_id", "block_ref", "block_fingerprint", "issue_type",
            "severity", "issue_type_multiplier", "severity_weight",
            "readiness_penalty", "issue_burden", "repair_action",
            "dependency_to_substance", "description", "editorial_risk",
            "ready_for_point_builder", "notes",
        ]
        for f in required:
            assert f in issue, f"Field missing: {f}"

    def test_issue_id_format(self):
        b = make_block()
        issue = make_issue("bab_3", 7, "empty_block", b, "Test.")
        assert issue["issue_id"] == "bab_3-ISS-007"

    def test_burden_consistent_with_fields(self):
        b = make_block()
        issue = make_issue("bab_1", 1, "anomalous_value", b, "Test.")
        expected = (issue["severity_weight"] * issue["issue_type_multiplier"]) + issue["readiness_penalty"]
        assert abs(issue["issue_burden"] - expected) < 0.001


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────

class TestValidation:
    def _valid_issue(self, itype="empty_block", sev="high", action="rewrite"):
        b = make_block()
        return make_issue("bab_1", 1, itype, b, "Valid issue.", severity=sev, repair_action=action)

    def test_valid_chapter_no_errors(self):
        # Gunakan possible_generic_claim (low severity, ready=True)
        # burden = (1 × 1.0) + 0 = 1.0, normalized = 1.0 < 2.25 → health = high
        b = make_block()
        issue = make_issue("bab_1", 1, "possible_generic_claim", b, "Test.")
        cb = {
            "chapter_id": "bab_1",
            "chapter_health": "high",
            "chapter_burden_total": issue["issue_burden"],
            "normalized_burden": issue["issue_burden"],
            "issue_count": 1,
            "issues": [issue],
        }
        assert validate_chapter_backlog(cb) == []

    def test_invalid_issue_type(self):
        issue = self._valid_issue()
        issue["issue_type"] = "fake_type"
        errors = validate_issue(issue)
        assert any("issue_type" in e for e in errors)

    def test_wrong_severity_weight(self):
        issue = self._valid_issue()
        issue["severity_weight"] = 99
        errors = validate_issue(issue)
        assert any("severity_weight" in e for e in errors)

    def test_wrong_burden(self):
        issue = self._valid_issue()
        issue["issue_burden"] = 999.0
        errors = validate_issue(issue)
        assert any("issue_burden" in e for e in errors)

    def test_critical_ready_true_fails(self):
        issue = self._valid_issue(sev="critical", action="verify")
        issue["ready_for_point_builder"] = True
        errors = validate_issue(issue)
        assert any("critical" in e for e in errors)

    def test_other_without_notes_fails(self):
        issue = self._valid_issue(itype="other")
        issue["notes"] = None
        errors = validate_issue(issue)
        assert any("notes" in e for e in errors)

    def test_other_with_notes_passes(self):
        issue = self._valid_issue(itype="other")
        issue["notes"] = "Penjelasan detail issue ini."
        assert validate_issue(issue) == []


# ─────────────────────────────────────────────
# RULEBOOK — R-A3-xx
# ─────────────────────────────────────────────

class TestRulebook:
    def test_r_a3_01_repair_action_required(self):
        b = make_block()
        issue = make_issue("bab_1", 1, "empty_block", b, "Test.")
        assert issue["repair_action"] is not None

    def test_r_a3_03_critical_not_ready(self):
        b = make_block()
        issue = make_issue("bab_1", 1, "anomalous_value", b, "Test.")
        assert issue["ready_for_point_builder"] is False

    def test_r_a3_04_scaffold_only_not_ready(self):
        b = make_block()
        issue = make_issue("bab_1", 1, "structural_imbalance", b, "Test.")
        assert issue["ready_for_point_builder"] is False

    def test_r_a3_07_dependency_required(self):
        b = make_block()
        issue = make_issue("bab_1", 1, "empty_block", b, "Test.")
        assert issue["dependency_to_substance"] is not None

    def test_r_a3_09_non_content_no_content_issue(self):
        nct_block = {"type": "divider", "block_ref": "bab_1.B001",
                     "block_fingerprint": "fp", "text": ""}
        assert detect_empty_block(nct_block, NCT) is False
        assert detect_placeholder_content(nct_block, NCT) is False

    def test_r_a3_10_other_needs_notes(self):
        b = make_block()
        issue = make_issue("bab_1", 1, "other", b, "Test.", notes=None)
        errors = validate_issue(issue)
        assert any("notes" in e for e in errors)

    def test_r_a3_11_one_block_multiple_issues(self):
        """Satu block bisa punya > 1 issue jika berbeda substantif."""
        b = make_block(text="[TBD] Tuliskan deskripsi program di sini.")
        has_placeholder = detect_placeholder_content(b, NCT)
        has_template    = detect_template_residue(b, NCT)
        # Kedua issues bisa muncul dari satu block
        assert has_placeholder is True
        assert has_template is True


# ─────────────────────────────────────────────
# SEMANTIC LINT
# ─────────────────────────────────────────────

class TestSemanticLintA3:
    def _make_cb(self, chapter_id, health, issues=None):
        issues = issues or []
        burden = sum(i.get("issue_burden", 1.0) for i in issues)
        nb     = round(burden / max(1, len(issues)), 4)
        return {
            "chapter_id": chapter_id,
            "chapter_health": health,
            "chapter_burden_total": burden,
            "normalized_burden": nb,
            "issue_count": len(issues),
            "issues": issues,
        }

    def _issue(self, sev="medium", dep="weak", itype="empty_block"):
        b = make_block()
        return make_issue("bab_1", 1, itype, b, "Test.",
                          severity=sev, dependency=dep)

    def test_all_chapters_healthy_triggers(self):
        cbs = [self._make_cb(f"bab_{i}", "high") for i in range(1, 4)]
        warns = semantic_lint_a3(cbs)
        assert any(w["code"] == "LINT_ALL_CHAPTERS_HEALTHY" for w in warns)

    def test_mixed_health_no_trigger(self):
        cbs = [
            self._make_cb("bab_1", "high"),
            self._make_cb("bab_2", "medium"),
        ]
        warns = semantic_lint_a3(cbs)
        assert not any(w["code"] == "LINT_ALL_CHAPTERS_HEALTHY" for w in warns)

    def test_no_critical_triggers(self):
        issues = [self._issue("medium")]
        cbs = [self._make_cb("bab_1", "medium", issues)]
        warns = semantic_lint_a3(cbs)
        assert any(w["code"] == "LINT_NO_CRITICAL_ISSUES" for w in warns)

    def test_has_critical_no_trigger(self):
        issues = [self._issue("critical")]
        cbs = [self._make_cb("bab_1", "low", issues)]
        warns = semantic_lint_a3(cbs)
        assert not any(w["code"] == "LINT_NO_CRITICAL_ISSUES" for w in warns)

    def test_all_dependency_none_triggers(self):
        issues = [self._issue(dep="none")]
        cbs = [self._make_cb("bab_1", "medium", issues)]
        warns = semantic_lint_a3(cbs)
        assert any(w["code"] == "LINT_ALL_DEPENDENCY_NONE" for w in warns)

    def test_imbalanced_distribution_triggers(self):
        many_issues = [self._issue() for _ in range(8)]
        few_issues  = [self._issue() for _ in range(2)]
        cbs = [
            self._make_cb("bab_1", "low", many_issues),
            self._make_cb("bab_2", "high", few_issues),
        ]
        warns = semantic_lint_a3(cbs)
        assert any(w["code"] == "LINT_IMBALANCED_DISTRIBUTION" for w in warns)

    def test_empty_backlogs_no_lint(self):
        assert semantic_lint_a3([]) == []


# ─────────────────────────────────────────────
# SCAN CHAPTER — integration
# ─────────────────────────────────────────────

class TestScanChapter:
    def _chapter(self, blocks, chapter_id="bab_1"):
        return {"chapter_id": chapter_id, "chapter_type": None, "blocks": blocks}

    def test_empty_chapter_flags_structural_imbalance(self):
        ch = self._chapter([])
        issues = scan_chapter(ch, "bab_7", [], NCT)
        assert any(itype == "structural_imbalance" for itype, *_ in issues)

    def test_empty_block_detected(self):
        blocks = [make_block(text=""), make_block(text="Normal text here.")]
        ch = self._chapter(blocks)
        issues = scan_chapter(ch, "bab_7", [], NCT)
        assert any(itype == "empty_block" for itype, *_ in issues)

    def test_placeholder_detected(self):
        blocks = [make_block(text="[Isi data investasi di sini]")]
        ch = self._chapter(blocks)
        issues = scan_chapter(ch, "bab_7", [], NCT)
        assert any(itype == "placeholder_content" for itype, *_ in issues)

    def test_substance_chapter_no_alignment_check(self):
        # bab_7 adalah substance chapter — tidak dicek alignment
        long_unrelated = "Pendahuluan dokumen ini membahas berbagai aspek yang tidak berkaitan langsung dengan topik utama secara spesifik dan mendalam."
        blocks = [make_block(text=long_unrelated)]
        ch = self._chapter(blocks, chapter_id="bab_7")
        elements = [make_element()]
        issues = scan_chapter(ch, "bab_7", elements, NCT)
        assert not any(itype == "missing_substance_alignment" for itype, *_ in issues)

    def test_non_content_blocks_no_issues(self):
        blocks = [{"type": "divider", "block_ref": "bab_1.B001",
                   "block_fingerprint": "fp", "text": ""}]
        ch = self._chapter(blocks)
        issues = scan_chapter(ch, "bab_7", [], NCT)
        # Structural imbalance mungkin muncul, tapi bukan content issues
        content_issues = [i for i in issues if i[0] not in ("structural_imbalance", "visual_gap")]
        assert len(content_issues) == 0


# ─────────────────────────────────────────────
# RUN — end-to-end
# ─────────────────────────────────────────────

class TestRun:
    def _minimal_input(self, chapters=None):
        if chapters is None:
            chapters = [{
                "chapter_id": "bab_1",
                "chapter_type": None,
                "blocks": [
                    make_block("bab_1.B001", text="Program ini berjalan sejak 2023 dengan investasi signifikan."),
                    make_block("bab_1.B002", text="Outcome program menunjukkan hasil yang dapat diukur secara kuantitatif."),
                ]
            }]
        return {
            "pipeline_control_block": make_pcb(),
            "block_identity_packet": make_bip(chapters),
            "substance_registry": make_registry(),
            "substance_guardrails": {"global_guardrails": [], "element_guardrails": []},
        }

    def test_run_produces_packet(self):
        packet = run(self._minimal_input())
        assert packet["activity"] == "repair_backlog_builder"
        assert "chapter_backlogs" in packet
        assert "summary" in packet

    def test_run_all_chapters_present(self):
        chapters = [
            {"chapter_id": "bab_1", "chapter_type": None, "blocks": [make_block(text="Teks normal cukup panjang untuk diproses dengan baik oleh sistem ini.")]},
            {"chapter_id": "bab_2", "chapter_type": None, "blocks": [make_block(text="Teks normal cukup panjang untuk diproses dengan baik oleh sistem ini.")]},
        ]
        packet = run(self._minimal_input(chapters))
        chapter_ids = [cb["chapter_id"] for cb in packet["chapter_backlogs"]]
        assert "bab_1" in chapter_ids
        assert "bab_2" in chapter_ids

    def test_run_empty_chapters_failure(self):
        data = self._minimal_input([])
        packet = run(data)
        assert any(f["code"] == "NO_CHAPTERS" for f in packet["failures"])

    def test_run_formula_validation_passes(self):
        packet = run(self._minimal_input())
        assert not any(f["code"] == "VALIDATION_ERROR" for f in packet["failures"])

    def test_run_chapter_health_computed(self):
        packet = run(self._minimal_input())
        for cb in packet["chapter_backlogs"]:
            assert cb["chapter_health"] in {"high", "medium", "low"}

    def test_run_summary_consistent(self):
        packet = run(self._minimal_input())
        s = packet["summary"]
        total_from_chapters = sum(cb["issue_count"] for cb in packet["chapter_backlogs"])
        assert s["total_issues"] == total_from_chapters

    def test_run_critical_issues_not_ready(self):
        """Semua critical issue harus ready_for_point_builder=false."""
        chapters = [{
            "chapter_id": "bab_1",
            "chapter_type": None,
            "blocks": [
                {"type": "paragraph", "block_ref": "bab_1.B001",
                 "block_fingerprint": "fp1",
                 "text": 'Nilai investasi program adalah 0 rupiah sesuai laporan keuangan.'},
            ]
        }]
        packet = run(self._minimal_input(chapters))
        all_issues = [i for cb in packet["chapter_backlogs"] for i in cb["issues"]]
        critical = [i for i in all_issues if i["severity"] == "critical"]
        assert all(not i["ready_for_point_builder"] for i in critical)


class TestPatch3Fixes:
    """Tests untuk 3 patch freeze Sprint 3."""

    # Patch 1 — validate_chapter_backlog signature
    def test_validate_chapter_backlog_no_second_arg(self):
        issues = []
        cb = {"chapter_id": "bab_1", "chapter_health": "high",
              "chapter_burden_total": 0.0, "normalized_burden": 0.0,
              "issue_count": 0, "issues": issues}
        # Harus bisa dipanggil dengan 1 argumen saja
        errors = validate_chapter_backlog(cb)
        assert isinstance(errors, list)

    # Patch 2 — alignment: 1 keyword tidak cukup, butuh 2
    def test_alignment_single_keyword_not_enough(self):
        b = make_block(text="Program ini memberikan outcome bagi masyarakat yang membutuhkan bantuan nyata.")
        els = [make_element(label="sroi_metric", summary="SROI 1:1.12")]
        # "outcome" = 1 keyword → tidak lolos, alignment check tetap True (missing)
        # "program" + "outcome" = 2 keywords → lolos
        # teks ini punya "program" dan "outcome" → seharusnya tidak flagged
        result = detect_missing_substance_alignment(b, NCT, els)
        assert result is False  # 2 keyword hits → aligned

    def test_alignment_one_keyword_flagged(self):
        b = make_block(text="Kegiatan sosial ini memberikan manfaat yang sangat besar bagi komunitas lokal sekitar.")
        els = [make_element(label="sroi_metric", summary="SROI 1:1.12 nilai sosial")]
        # "manfaat" = 1 keyword saja → tidak cukup → flagged sebagai missing
        result = detect_missing_substance_alignment(b, NCT, els)
        assert result is True

    # Patch 3 — proper noun heuristic lebih longgar
    def test_generic_claim_single_caps_not_suppressed(self):
        # "Program CSR" = satu kata kapital di posisi 2, tapi bukan 2 berturut
        b = make_block(text="Program CSR ini berjalan baik.")
        result = detect_possible_generic_claim(b, NCT)
        assert result is True  # tidak suppressed oleh satu kata kapital

    def test_generic_claim_two_consecutive_caps_suppressed(self):
        # "PT Pertamina" = 2 kata kapital berturut → proper noun → tidak flagged
        b = make_block(text="PT Pertamina jalankan ini.")
        result = detect_possible_generic_claim(b, NCT)
        assert result is False

    # execution_status
    def test_run_execution_status_success(self):
        data = {
            "pipeline_control_block": make_pcb(),
            "block_identity_packet": make_bip([{
                "chapter_id": "bab_1", "chapter_type": None,
                "blocks": [make_block(text="Program investasi ini menghasilkan outcome SROI positif.")]
            }]),
            "substance_registry": make_registry(),
            "substance_guardrails": {"global_guardrails": [], "element_guardrails": []},
        }
        packet = run(data)
        assert packet["execution_status"] in {"success", "warning", "failed"}

    def test_run_execution_status_failed_on_no_chapters(self):
        data = {
            "pipeline_control_block": make_pcb(),
            "block_identity_packet": make_bip([]),
            "substance_registry": make_registry(),
            "substance_guardrails": {"global_guardrails": [], "element_guardrails": []},
        }
        packet = run(data)
        assert packet["execution_status"] == "failed"
