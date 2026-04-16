"""
A3 — Repair Backlog Builder
Sprint 3 | Deterministik — tidak menggunakan LLM.

R: (D, S, G) → B
  D = block_identity_packet
  S = substance_registry
  G = substance_guardrails
  B = repair_backlog_packet

Dua kelas issue:
  I_d = hard deterministic issues
  I_h = heuristic flags (prefix possible_)
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ACTIVITY         = "repair_backlog_builder"
PIPELINE_VERSION = "v1"

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

SEVERITY_WEIGHT = {"low": 1, "medium": 2, "high": 4, "critical": 7}

ISSUE_TYPE_MULTIPLIER: dict[str, float] = {
    "empty_block":                     1.5,
    "placeholder_content":             1.5,
    "anomalous_value":                 1.5,
    "possible_unsupported_inference":  1.25,
    "evidence_gap":                    1.25,
}

DEFAULT_SEVERITY: dict[str, str] = {
    "empty_block":                    "high",
    "empty_table":                    "high",
    "empty_list":                     "medium",
    "placeholder_content":            "high",
    "template_residue":               "medium",
    "anomalous_value":                "critical",
    "structural_imbalance":           "medium",
    "visual_gap":                     "low",
    "missing_substance_alignment":    "medium",
    "evidence_gap":                   "medium",
    "duplicate_function":             "low",
    "possible_generic_claim":         "low",
    "possible_unsupported_inference": "medium",
    "possible_weak_transition":       "low",
    "possible_overclaim_risk":        "medium",
    "other":                          "medium",
}

DEFAULT_REPAIR_ACTION: dict[str, str] = {
    "empty_block":                    "rewrite",
    "empty_table":                    "rewrite",
    "empty_list":                     "rewrite",
    "placeholder_content":            "rewrite",
    "template_residue":               "remove",
    "anomalous_value":                "verify",
    "structural_imbalance":           "scaffold_only",
    "visual_gap":                     "scaffold_only",
    "missing_substance_alignment":    "refine",
    "evidence_gap":                   "verify",
    "duplicate_function":             "remove",
    "possible_generic_claim":         "refine",
    "possible_unsupported_inference": "scaffold_only",
    "possible_weak_transition":       "refine",
    "possible_overclaim_risk":        "verify",
    "other":                          "refine",
}

DEFAULT_DEPENDENCY: dict[str, str] = {
    "missing_substance_alignment":    "strong",
    "evidence_gap":                   "strong",
    "possible_overclaim_risk":        "medium",
    "possible_unsupported_inference": "medium",
    "possible_generic_claim":         "weak",
    "possible_weak_transition":       "none",
    "empty_block":                    "weak",
    "empty_table":                    "weak",
    "empty_list":                     "none",
    "placeholder_content":            "weak",
    "template_residue":               "none",
    "anomalous_value":                "medium",
    "structural_imbalance":           "none",
    "visual_gap":                     "weak",
    "duplicate_function":             "none",
    "other":                          "weak",
}

# Chapters yang relevan untuk visual_gap check
VISUAL_RELEVANT_CHAPTERS = {"bab_4", "bab_5", "bab_6", "bab_7", "bab_8"}

# Block types yang dianggap visual-capable
VISUAL_BLOCK_TYPES = {
    "table", "chart", "metric_card", "metric_card_3col", "metric_card_2col",
    "infographic", "diagram", "bar_chart", "line_chart", "pie_chart",
}

# Valid enums
VALID_ISSUE_TYPES = {
    "empty_block", "empty_table", "empty_list", "placeholder_content",
    "template_residue", "anomalous_value", "structural_imbalance",
    "visual_gap", "missing_substance_alignment", "evidence_gap",
    "duplicate_function", "possible_generic_claim",
    "possible_unsupported_inference", "possible_weak_transition",
    "possible_overclaim_risk", "other",
}
VALID_SEVERITIES        = {"low", "medium", "high", "critical"}
VALID_REPAIR_ACTIONS    = {"refine", "rewrite", "scaffold_only", "hold", "remove", "verify"}
VALID_DEPENDENCIES      = {"none", "weak", "medium", "strong"}
VALID_EDITORIAL_RISKS   = {"low", "medium", "high"}
CHAPTER_LEVEL_ISSUES    = {"visual_gap", "structural_imbalance"}

# Heuristic patterns
RE_PLACEHOLDER = re.compile(
    r'(\[.*?\]|TBD|TODO|lorem ipsum|PLACEHOLDER|\(isi di sini\)|\.\.\.|FIXME)',
    re.IGNORECASE,
)
RE_TEMPLATE_RESIDUE = re.compile(
    r'\b(tuliskan|jelaskan|masukkan|deskripsikan|isi dengan|tambahkan|sebutkan)\b',
    re.IGNORECASE,
)
RE_INFERENTIAL_START = re.compile(
    r'^(terbukti|menunjukkan|menghasilkan|membuktikan|menegaskan|memperlihatkan)\b',
    re.IGNORECASE,
)
RE_CAVEAT = re.compile(
    r'\b(mungkin|diperkirakan|proxy|estimasi|diduga|kemungkinan|berpotensi)\b',
    re.IGNORECASE,
)
RE_HAS_NUMBER = re.compile(r'\d')
RE_ANOMALOUS_CONTEXTS = re.compile(
    r'\b(investasi|investment|sroi|beneficiary|outcome_value|penerima|manfaat)\b',
    re.IGNORECASE,
)
ANOMALOUS_SENTINELS = {"999999", "-1", "nan", "null", "none", "0"}

# SROI substantive keywords untuk generic_claim detection
SROI_KEYWORDS = {
    "sroi", "investasi", "outcome", "dampak", "manfaat", "monetisasi",
    "nilai sosial", "beneficiary", "penerima", "evaluasi",
    "indikator", "baseline", "intervensi", "kausalitas", "proxy",
    "observed", "ddat", "adjustment", "haircut",
}


# ─────────────────────────────────────────────
# BURDEN & READINESS
# ─────────────────────────────────────────────

def compute_readiness(severity: str, issue_type: str, repair_action: str) -> bool:
    """R-A3-03, R-A3-04: critical dan beberapa repair_action → not ready."""
    if severity == "critical":
        return False
    if repair_action in {"scaffold_only", "verify", "hold"}:
        return False
    return True


def compute_issue_burden(
    severity: str,
    issue_type: str,
    ready_for_point_builder: bool,
) -> float:
    sw  = SEVERITY_WEIGHT.get(severity, 1)
    itm = ISSUE_TYPE_MULTIPLIER.get(issue_type, 1.0)
    rp  = 0 if ready_for_point_builder else 2
    return round((sw * itm) + rp, 4)


def compute_chapter_health(
    issues: list[dict],
    normalized_burden: float,
) -> str:
    """
    Formula resmi + override rules (R-A3-06).
    Override bersifat monotonic — hanya bisa turunkan health.
    """
    # Base dari formula
    if normalized_burden < 2.25:
        health = "high"
    elif normalized_burden < 4.5:
        health = "medium"
    else:
        health = "low"

    if health == "low":
        return "low"

    # Override conditions
    critical_count = sum(1 for i in issues if i["severity"] == "critical")
    if critical_count >= 2:
        return "low"

    anomalous_not_ready = any(
        i["issue_type"] == "anomalous_value" and not i["ready_for_point_builder"]
        for i in issues
    )
    if anomalous_not_ready:
        return "low"

    if issues:
        not_ready_ratio = sum(1 for i in issues if not i["ready_for_point_builder"]) / len(issues)
        if not_ready_ratio > 0.3:
            return "low"

    return health


# ─────────────────────────────────────────────
# ISSUE BUILDER
# ─────────────────────────────────────────────

def make_issue(
    chapter_id: str,
    seq: int,
    issue_type: str,
    block: dict | None,
    description: str,
    notes: str | None = None,
    severity: str | None = None,
    repair_action: str | None = None,
    dependency: str | None = None,
    editorial_risk: str | None = None,
) -> dict:
    """Bangun satu issue dengan semua field lengkap."""
    sev    = severity      or DEFAULT_SEVERITY.get(issue_type, "medium")
    action = repair_action or DEFAULT_REPAIR_ACTION.get(issue_type, "refine")
    dep    = dependency    or DEFAULT_DEPENDENCY.get(issue_type, "none")
    erisk  = editorial_risk or _default_editorial_risk(sev)

    # Chapter-level issues: block_ref dan block_fingerprint = null
    if issue_type in CHAPTER_LEVEL_ISSUES or block is None:
        block_ref = None
        block_fp  = None
    else:
        block_ref = block.get("block_ref")
        block_fp  = block.get("block_fingerprint")

    ready   = compute_readiness(sev, issue_type, action)
    burden  = compute_issue_burden(sev, issue_type, ready)
    sw      = SEVERITY_WEIGHT.get(sev, 1)
    itm     = ISSUE_TYPE_MULTIPLIER.get(issue_type, 1.0)
    rp      = 0 if ready else 2

    return {
        "issue_id":               f"{chapter_id}-ISS-{seq:03d}",
        "block_ref":              block_ref,
        "block_fingerprint":      block_fp,
        "issue_type":             issue_type,
        "severity":               sev,
        "issue_type_multiplier":  itm,
        "severity_weight":        sw,
        "readiness_penalty":      rp,
        "issue_burden":           burden,
        "repair_action":          action,
        "dependency_to_substance": dep,
        "description":            description,
        "editorial_risk":         erisk,
        "ready_for_point_builder": ready,
        "notes":                  notes,
    }


def _default_editorial_risk(severity: str) -> str:
    return {"low": "low", "medium": "medium", "high": "high", "critical": "high"}.get(severity, "medium")


# ─────────────────────────────────────────────
# DETECTION — HARD DETERMINISTIC
# ─────────────────────────────────────────────

def detect_empty_block(block: dict, nct: set) -> bool:
    """Block non-non-content dengan text kosong."""
    if block.get("type") in nct:
        return False
    text = (block.get("text") or "").strip()
    return not text


def detect_empty_table(block: dict) -> bool:
    if block.get("type") != "table":
        return False
    rows = block.get("rows")
    return not rows or rows == []


def detect_empty_list(block: dict) -> bool:
    if block.get("type") != "list":
        return False
    items = block.get("items")
    return not items or items == []


def detect_placeholder_content(block: dict, nct: set) -> bool:
    if block.get("type") in nct:
        return False
    text = (block.get("text") or "").strip()
    if not text:
        return False
    return bool(RE_PLACEHOLDER.search(text))


def detect_template_residue(block: dict, nct: set) -> bool:
    if block.get("type") in nct:
        return False
    text = (block.get("text") or "").strip()
    if not text:
        return False
    return bool(RE_TEMPLATE_RESIDUE.search(text))


def detect_anomalous_value(block: dict, nct: set) -> bool:
    """
    Pattern-based: nilai 0/sentinel pada konteks investment/sroi/beneficiary,
    atau nilai sentinel universal.
    """
    if block.get("type") in nct:
        return False
    text = (block.get("text") or "").strip().lower()
    if not text:
        return False

    # Nilai sentinel universal
    for sentinel in ANOMALOUS_SENTINELS - {"0"}:
        if sentinel in text.split():
            return True

    # Angka 0 pada konteks sensitif
    if "0" in text and RE_ANOMALOUS_CONTEXTS.search(text):
        # Cek apakah "0" muncul sebagai angka (bukan bagian dari angka lain)
        if re.search(r'\b0\b', text):
            return True

    # String null/none terserialisasi
    if re.search(r'"null"|"none"|\'null\'|\'none\'', text, re.IGNORECASE):
        return True

    return False


def detect_duplicate_function(blocks: list[dict], nct: set) -> list[tuple[dict, dict]]:
    """
    Temukan pasangan block dalam chapter yang punya fingerprint prefix identik.
    Returns list of (block_a, block_b) pairs.
    """
    seen: dict[str, dict] = {}
    duplicates = []
    for block in blocks:
        if block.get("type") in nct:
            continue
        fp = block.get("block_fingerprint", "")
        # Gunakan seluruh fingerprint (sudah mencerminkan normalized prefix)
        if fp in seen:
            duplicates.append((seen[fp], block))
        else:
            seen[fp] = block
    return duplicates


def detect_missing_substance_alignment(
    block: dict,
    nct: set,
    substance_elements: list[dict],
) -> bool:
    """
    Cek apakah block tidak punya lexical/structural anchor ke substance registry.
    Hanya untuk non-substance chapters, hanya untuk content blocks >= 20 kata.
    """
    if block.get("type") in nct:
        return False

    text = (block.get("text") or "").strip()
    words = text.split()
    if len(words) < 10:
        return False

    text_lower = text.lower()

    for el in substance_elements:
        # 1. Lexical overlap dengan label atau summary
        label_words = [w for w in re.split(r'[_\s]+', el.get("label", "")) if len(w) > 4]
        for word in label_words:
            if word.lower() in text_lower:
                return False

        summary = el.get("summary", "").lower()
        summary_words = [w for w in summary.split() if len(w) > 4]
        overlap = sum(1 for w in summary_words if w in text_lower)
        if overlap >= 2:
            return False

        # 2. Numeric anchor
        numbers_in_summary = set(re.findall(r'\b\d[\d,.]*\b', summary))
        numbers_in_text    = set(re.findall(r'\b\d[\d,.]*\b', text_lower))
        if numbers_in_summary & numbers_in_text:
            return False

        # 3. Affordance anchor keywords
        affordance_keywords = {
            "outcome", "investasi", "sroi", "monetisasi", "dampak",
            "manfaat", "beneficiary", "program", "evaluasi", "substansi",
        }
        for kw in affordance_keywords:
            if kw in text_lower:
                return False

    return True  # tidak ada alignment


def detect_evidence_gap(
    block: dict,
    nct: set,
    substance_elements: list[dict],
) -> bool:
    """
    Block merujuk elemen registry yang evidence_status=pending.
    """
    if block.get("type") in nct:
        return False
    text = (block.get("text") or "").strip().lower()
    if not text:
        return False

    for el in substance_elements:
        if el.get("evidence_status") != "pending":
            continue
        label_words = [w for w in re.split(r'[_\s]+', el.get("label", "")) if len(w) > 4]
        for word in label_words:
            if word.lower() in text:
                return True

    return False


# ─────────────────────────────────────────────
# DETECTION — HEURISTIC FLAGS
# ─────────────────────────────────────────────

def detect_possible_generic_claim(block: dict, nct: set) -> bool:
    if block.get("type") in nct:
        return False
    text = (block.get("text") or "").strip()
    if not text:
        return False
    words = text.split()
    if len(words) >= 15:
        return False
    if RE_HAS_NUMBER.search(text):
        return False
    # Cek proper noun — kata dengan huruf kapital di tengah kalimat
    has_proper_noun = any(
        w[0].isupper() for w in words[1:] if len(w) > 2
    )
    if has_proper_noun:
        return False
    # Cek SROI keywords
    text_lower = text.lower()
    has_substantive = any(kw in text_lower for kw in SROI_KEYWORDS)
    if has_substantive:
        return False
    return True


def detect_possible_unsupported_inference(block: dict, nct: set) -> bool:
    if block.get("type") in nct:
        return False
    text = (block.get("text") or "").strip()
    if not text:
        return False
    if not RE_INFERENTIAL_START.match(text):
        return False
    # Ada angka → ada data pendukung
    if RE_HAS_NUMBER.search(text):
        return False
    # Ada referensi eksplisit
    if re.search(r'\b(lihat|merujuk|berdasarkan|sesuai|mengacu|data|tabel|grafik)\b', text, re.IGNORECASE):
        return False
    return True


def detect_possible_weak_transition(block: dict, nct: set, is_first_or_last: bool) -> bool:
    if not is_first_or_last:
        return False
    if block.get("type") in nct:
        return False
    text = (block.get("text") or "").strip()
    words = text.split()
    return len(words) < 6


def detect_possible_overclaim_risk(
    block: dict,
    nct: set,
    substance_elements: list[dict],
) -> bool:
    """
    Block mengandung lexical overlap dengan elemen proxy
    tapi tidak ada caveat hedging.
    """
    if block.get("type") in nct:
        return False
    text = (block.get("text") or "").strip()
    if not text:
        return False

    proxy_elements = [
        el for el in substance_elements
        if el.get("evidence_status") in {"proxy", "inferred"}
    ]
    if not proxy_elements:
        return False

    text_lower = text.lower()
    has_overlap = False
    for el in proxy_elements:
        label_words = [w for w in re.split(r'[_\s]+', el.get("label", "")) if len(w) > 4]
        for word in label_words:
            if word.lower() in text_lower:
                has_overlap = True
                break

    if not has_overlap:
        return False

    # Ada caveat → tidak overclaim
    if RE_CAVEAT.search(text):
        return False

    return True


# ─────────────────────────────────────────────
# CHAPTER-LEVEL DETECTORS
# ─────────────────────────────────────────────

def detect_structural_imbalance(content_blocks: list[dict]) -> bool:
    """Chapter punya < 2 content blocks."""
    return len(content_blocks) < 2


def detect_visual_gap(
    chapter_id: str,
    blocks: list[dict],
    substance_elements: list[dict],
) -> bool:
    """
    Chapter termasuk visual-relevant AND tidak punya visual block
    AND substance registry punya visual_candidate affordance.
    """
    if chapter_id not in VISUAL_RELEVANT_CHAPTERS:
        return False

    has_visual = any(b.get("type") in VISUAL_BLOCK_TYPES for b in blocks)
    if has_visual:
        return False

    has_visual_candidate = any(
        "visual_candidate" in el.get("use_affordances", [])
        for el in substance_elements
    )
    return has_visual_candidate


# ─────────────────────────────────────────────
# CHAPTER SCANNER
# ─────────────────────────────────────────────

def scan_chapter(
    chapter: dict,
    substance_chapter_id: str,
    substance_elements: list[dict],
    nct: set,
) -> list[dict]:
    """
    Scan satu chapter dan hasilkan semua candidate issues.
    Returns list of issue dicts (belum punya issue_id).
    """
    chapter_id    = chapter.get("chapter_id", "?")
    blocks        = chapter.get("blocks", [])
    content_blocks = [b for b in blocks if b.get("type") not in nct]
    is_substance  = (chapter_id == substance_chapter_id)
    issues_raw    = []  # list of (issue_type, block|None, description, notes, overrides)

    # ── Chapter-level checks ─────────────────
    if detect_structural_imbalance(content_blocks):
        issues_raw.append((
            "structural_imbalance", None,
            f"Chapter '{chapter_id}' hanya punya {len(content_blocks)} content block(s). "
            "Struktur bab terlalu tipis untuk membangun narasi.",
            None, {},
        ))

    if detect_visual_gap(chapter_id, blocks, substance_elements):
        issues_raw.append((
            "visual_gap", None,
            f"Chapter '{chapter_id}' tidak punya visual block (tabel/grafik/metric card) "
            "meskipun substance registry punya elemen visual_candidate.",
            None, {},
        ))

    # ── Block-level checks ───────────────────
    seen_fingerprints: dict[str, dict] = {}
    first_content_idx = None
    last_content_idx  = None

    # Tentukan first/last content block index
    for i, block in enumerate(blocks):
        if block.get("type") not in nct:
            if first_content_idx is None:
                first_content_idx = i
            last_content_idx = i

    for i, block in enumerate(blocks):
        btype = block.get("type", "")
        if btype in nct:
            continue

        is_boundary = (i == first_content_idx or i == last_content_idx)

        # Hard: empty_block
        if detect_empty_block(block, nct):
            issues_raw.append((
                "empty_block", block,
                f"Block {block.get('block_ref')} tidak memiliki konten teks.",
                None, {},
            ))
            continue  # kosong → tidak perlu cek lain

        # Hard: empty_table
        if detect_empty_table(block):
            issues_raw.append((
                "empty_table", block,
                f"Block {block.get('block_ref')} adalah tabel tanpa baris data.",
                None, {},
            ))

        # Hard: empty_list
        if detect_empty_list(block):
            issues_raw.append((
                "empty_list", block,
                f"Block {block.get('block_ref')} adalah list tanpa item.",
                None, {},
            ))

        # Hard: placeholder_content
        if detect_placeholder_content(block, nct):
            issues_raw.append((
                "placeholder_content", block,
                f"Block {block.get('block_ref')} mengandung teks placeholder yang belum diganti.",
                None, {},
            ))

        # Hard: template_residue
        if detect_template_residue(block, nct):
            issues_raw.append((
                "template_residue", block,
                f"Block {block.get('block_ref')} mengandung instruksi template yang tidak dihapus.",
                None, {},
            ))

        # Hard: anomalous_value
        if detect_anomalous_value(block, nct):
            issues_raw.append((
                "anomalous_value", block,
                f"Block {block.get('block_ref')} mengandung nilai anomalis "
                "(angka 0 pada konteks substantif, atau nilai sentinel).",
                "Kemungkinan placeholder atau error input data.",
                {},
            ))

        # Hard: duplicate_function — cek via fingerprint
        fp = block.get("block_fingerprint", "")
        if fp and fp in seen_fingerprints:
            issues_raw.append((
                "duplicate_function", block,
                f"Block {block.get('block_ref')} memiliki fingerprint identik dengan "
                f"{seen_fingerprints[fp].get('block_ref')} — kemungkinan konten duplikat.",
                None, {},
            ))
        elif fp:
            seen_fingerprints[fp] = block

        # Hard: missing_substance_alignment (hanya non-substance chapter)
        if not is_substance and detect_missing_substance_alignment(block, nct, substance_elements):
            issues_raw.append((
                "missing_substance_alignment", block,
                f"Block {block.get('block_ref')} tidak memiliki anchor lexical/struktural "
                "terhadap elemen di substance registry.",
                None, {"dependency_to_substance": "strong"},
            ))

        # Hard: evidence_gap
        if detect_evidence_gap(block, nct, substance_elements):
            issues_raw.append((
                "evidence_gap", block,
                f"Block {block.get('block_ref')} merujuk elemen substansi dengan "
                "evidence_status=pending yang belum terverifikasi.",
                None, {},
            ))

        # Heuristic: possible_generic_claim
        if detect_possible_generic_claim(block, nct):
            issues_raw.append((
                "possible_generic_claim", block,
                f"Block {block.get('block_ref')} berpotensi merupakan klaim generik: "
                "kalimat pendek tanpa angka, proper noun, atau keyword substantif.",
                "Heuristic flag — perlu review manual.",
                {},
            ))

        # Heuristic: possible_unsupported_inference
        if detect_possible_unsupported_inference(block, nct):
            issues_raw.append((
                "possible_unsupported_inference", block,
                f"Block {block.get('block_ref')} dimulai dengan kata inferensial "
                "tanpa angka atau referensi pendukung.",
                "Heuristic flag — perlu review manual.",
                {},
            ))

        # Heuristic: possible_weak_transition
        if detect_possible_weak_transition(block, nct, is_boundary):
            issues_raw.append((
                "possible_weak_transition", block,
                f"Block {block.get('block_ref')} berada di posisi batas chapter "
                "dan sangat pendek (< 8 kata) — potensi transisi lemah.",
                "Heuristic flag — perlu review manual.",
                {},
            ))

        # Heuristic: possible_overclaim_risk
        if not is_substance and detect_possible_overclaim_risk(block, nct, substance_elements):
            issues_raw.append((
                "possible_overclaim_risk", block,
                f"Block {block.get('block_ref')} mengandung klaim yang beroverlap "
                "dengan elemen proxy/inferred tanpa caveat hedging.",
                "Heuristic flag — perlu review manual.",
                {},
            ))

    return issues_raw


# ─────────────────────────────────────────────
# SEMANTIC LINT — V_meta(B)
# ─────────────────────────────────────────────

def semantic_lint_a3(chapter_backlogs: list[dict]) -> list[dict]:
    """
    Validasi meta-level atas perilaku backlog secara global.
    Deteksi pola suspicious yang lolos per-issue validation.
    """
    warnings = []
    if not chapter_backlogs:
        return warnings

    all_issues = [iss for cb in chapter_backlogs for iss in cb.get("issues", [])]

    # 1. Semua chapter health = high
    healths = [cb.get("chapter_health") for cb in chapter_backlogs]
    if all(h == "high" for h in healths) and len(healths) > 2:
        warnings.append({
            "code": "LINT_ALL_CHAPTERS_HEALTHY",
            "message": (
                f"Semua {len(healths)} chapter punya health='high'. "
                "Suspicious untuk dokumen SROI nyata — periksa apakah detection rules berjalan."
            ),
        })

    # 2. Tidak ada critical issue
    if all_issues and not any(i["severity"] == "critical" for i in all_issues):
        warnings.append({
            "code": "LINT_NO_CRITICAL_ISSUES",
            "message": (
                "Tidak ada issue critical sama sekali. "
                "Suspicious untuk dokumen yang belum sepenuhnya diverifikasi."
            ),
        })

    # 3. Semua dependency_to_substance = none
    if all_issues and all(i["dependency_to_substance"] == "none" for i in all_issues):
        warnings.append({
            "code": "LINT_ALL_DEPENDENCY_NONE",
            "message": (
                "Semua issue punya dependency_to_substance='none'. "
                "Tidak mungkin untuk dokumen SROI yang punya substansi registry."
            ),
        })

    # 4. Tidak ada heuristic flag sama sekali (untuk dokumen dengan banyak blocks)
    total_blocks = sum(cb.get("issue_count", 0) for cb in chapter_backlogs)
    heuristic_issues = [i for i in all_issues if i["issue_type"].startswith("possible_")]
    if total_blocks > 20 and not heuristic_issues:
        warnings.append({
            "code": "LINT_ZERO_HEURISTIC_FLAGS",
            "message": (
                f"Tidak ada heuristic flag (possible_*) dari {total_blocks} issues. "
                "Suspicious jika dokumen punya banyak content blocks."
            ),
        })

    # 5. Satu chapter dominasi > 60% total issues
    if all_issues:
        for cb in chapter_backlogs:
            chapter_issue_count = len(cb.get("issues", []))
            ratio = chapter_issue_count / len(all_issues)
            if ratio > 0.6 and len(all_issues) >= 5:
                warnings.append({
                    "code": "LINT_IMBALANCED_DISTRIBUTION",
                    "chapter_id": cb["chapter_id"],
                    "message": (
                        f"Chapter '{cb['chapter_id']}' mengandung {chapter_issue_count}/{len(all_issues)} "
                        f"({ratio:.0%}) dari total issues. Distribusi tidak seimbang."
                    ),
                })

    return warnings


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────

def validate_issue(issue: dict) -> list[str]:
    """Validasi satu issue terhadap semua V-A3-xx rules."""
    errors = []
    eid = issue.get("issue_id", "?")

    # V-A3-01: issue_type valid
    itype = issue.get("issue_type", "")
    if itype not in VALID_ISSUE_TYPES:
        errors.append(f"{eid}: issue_type tidak valid: '{itype}'")

    # V-A3-01: repair_action ada
    if not issue.get("repair_action"):
        errors.append(f"{eid}: repair_action wajib ada (R-A3-01)")

    # V-A3-02: severity_weight match
    sev = issue.get("severity", "")
    expected_sw = SEVERITY_WEIGHT.get(sev)
    if expected_sw is None:
        errors.append(f"{eid}: severity tidak valid: '{sev}'")
    elif issue.get("severity_weight") != expected_sw:
        errors.append(
            f"{eid}: severity_weight tidak match — expected {expected_sw}, "
            f"got {issue.get('severity_weight')}"
        )

    # V-A3-03: issue_type_multiplier match
    expected_itm = ISSUE_TYPE_MULTIPLIER.get(itype, 1.0)
    actual_itm   = issue.get("issue_type_multiplier")
    if actual_itm is not None and round(actual_itm, 4) != round(expected_itm, 4):
        errors.append(
            f"{eid}: issue_type_multiplier tidak match — expected {expected_itm}, "
            f"got {actual_itm}"
        )

    # V-A3-04: readiness_penalty
    ready = issue.get("ready_for_point_builder", True)
    expected_rp = 0 if ready else 2
    if issue.get("readiness_penalty") != expected_rp:
        errors.append(
            f"{eid}: readiness_penalty tidak match — expected {expected_rp}, "
            f"got {issue.get('readiness_penalty')}"
        )

    # V-A3-05: issue_burden formula
    sw  = issue.get("severity_weight", 1)
    itm = issue.get("issue_type_multiplier", 1.0)
    rp  = issue.get("readiness_penalty", 0)
    expected_burden = round((sw * itm) + rp, 4)
    actual_burden   = round(issue.get("issue_burden", 0), 4)
    if abs(actual_burden - expected_burden) > 0.001:
        errors.append(
            f"{eid}: issue_burden tidak match — expected {expected_burden}, "
            f"got {actual_burden}"
        )

    # V-A3-07: dependency_to_substance ada
    if issue.get("dependency_to_substance") not in VALID_DEPENDENCIES:
        errors.append(f"{eid}: dependency_to_substance tidak valid: '{issue.get('dependency_to_substance')}'")

    # V-A3-10: other → notes wajib non-empty
    if itype == "other" and not (issue.get("notes") or "").strip():
        errors.append(f"{eid}: issue_type=other wajib punya notes non-empty (R-A3-10)")

    # V-A3-10 (R-A3-03): critical → ready=false
    if sev == "critical" and issue.get("ready_for_point_builder") is True:
        errors.append(f"{eid}: severity=critical harus ready_for_point_builder=false (R-A3-03)")

    return errors


def validate_chapter_backlog(cb: dict, all_issues: list[dict]) -> list[str]:
    """Validasi satu chapter backlog."""
    errors = []
    chapter_id = cb.get("chapter_id", "?")
    issues     = cb.get("issues", [])

    # V-A3-06: chapter_burden_total
    expected_total = round(sum(i.get("issue_burden", 0) for i in issues), 4)
    actual_total   = round(cb.get("chapter_burden_total", 0), 4)
    if abs(actual_total - expected_total) > 0.01:
        errors.append(
            f"{chapter_id}: chapter_burden_total tidak match — "
            f"expected {expected_total}, got {actual_total}"
        )

    # V-A3-07: normalized_burden
    issue_count = max(1, len(issues))
    expected_nb = round(expected_total / issue_count, 4)
    actual_nb   = round(cb.get("normalized_burden", 0), 4)
    if abs(actual_nb - expected_nb) > 0.01:
        errors.append(
            f"{chapter_id}: normalized_burden tidak match — "
            f"expected {expected_nb}, got {actual_nb}"
        )

    # V-A3-08: chapter_health
    expected_health = compute_chapter_health(issues, actual_nb)
    if cb.get("chapter_health") != expected_health:
        errors.append(
            f"{chapter_id}: chapter_health tidak match — "
            f"expected '{expected_health}', got '{cb.get('chapter_health')}'"
        )

    return errors


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────

def run(input_data: dict) -> dict:
    """
    Entry point A3.
    input_data keys: pipeline_control_block, block_identity_packet,
                     substance_registry, substance_guardrails
    """
    pcb  = input_data.get("pipeline_control_block", {})
    bip  = input_data.get("block_identity_packet", {})
    reg  = input_data.get("substance_registry", {})
    _grd = input_data.get("substance_guardrails", {})  # tersedia untuk future use

    doc_id   = pcb.get("document_id", "unknown")
    prog     = pcb.get("program_code", "unknown")
    sub_ch   = pcb.get("substance_chapter_id", "")
    nct      = set(pcb.get("non_content_block_types", ["divider", "divider_thick", "spacer"]))

    substance_elements = reg.get("elements", [])
    chapters           = bip.get("chapters", [])
    now                = datetime.now(timezone.utc).isoformat()

    failures  = []
    warnings  = []
    chapter_backlogs = []

    if not chapters:
        failures.append({
            "code": "NO_CHAPTERS",
            "message": "block_identity_packet tidak mengandung chapters.",
        })

    for chapter in chapters:
        chapter_id   = chapter.get("chapter_id", "?")
        chapter_type = chapter.get("chapter_type")

        issues_raw = scan_chapter(chapter, sub_ch, substance_elements, nct)

        # Build issues dengan issue_id berurutan
        issues = []
        for seq, (itype, block, desc, notes, overrides) in enumerate(issues_raw, start=1):
            issue = make_issue(
                chapter_id=chapter_id,
                seq=seq,
                issue_type=itype,
                block=block,
                description=desc,
                notes=notes,
                severity=overrides.get("severity"),
                repair_action=overrides.get("repair_action"),
                dependency=overrides.get("dependency_to_substance"),
                editorial_risk=overrides.get("editorial_risk"),
            )
            issues.append(issue)

        # Aggregate
        burden_total = round(sum(i["issue_burden"] for i in issues), 4)
        issue_count  = len(issues)
        norm_burden  = round(burden_total / max(1, issue_count), 4)
        health       = compute_chapter_health(issues, norm_burden)

        chapter_backlogs.append({
            "chapter_id":          chapter_id,
            "chapter_type":        chapter_type,
            "chapter_health":      health,
            "chapter_burden_total": burden_total,
            "normalized_burden":   norm_burden,
            "issue_count":         issue_count,
            "issues":              issues,
        })

    # Validation
    all_errors = []
    for cb in chapter_backlogs:
        for issue in cb["issues"]:
            all_errors.extend(validate_issue(issue))
        all_errors.extend(validate_chapter_backlog(cb, cb["issues"]))

    # Semantic lint
    lint_warnings = semantic_lint_a3(chapter_backlogs)
    for lw in lint_warnings:
        warnings.append({"level": "warning", **lw})

    # Summary stats
    all_issues  = [i for cb in chapter_backlogs for i in cb["issues"]]
    by_type     = {}
    by_severity = {}
    by_health   = {}
    for i in all_issues:
        by_type[i["issue_type"]]   = by_type.get(i["issue_type"], 0) + 1
        by_severity[i["severity"]] = by_severity.get(i["severity"], 0) + 1
    for cb in chapter_backlogs:
        h = cb["chapter_health"]
        by_health[h] = by_health.get(h, 0) + 1

    # Fatal errors → failures
    if all_errors:
        for e in all_errors:
            failures.append({"code": "VALIDATION_ERROR", "message": e})

    packet = {
        "activity":               ACTIVITY,
        "document_id":            doc_id,
        "program_code":           prog,
        "generated_at":           now,
        "pipeline_version":       PIPELINE_VERSION,
        "chapter_backlogs":       chapter_backlogs,
        "document_level_findings": [],
        "summary": {
            "total_chapters":   len(chapter_backlogs),
            "total_issues":     len(all_issues),
            "issues_by_type":   by_type,
            "issues_by_severity": by_severity,
            "chapters_by_health": by_health,
            "failure_count":    len(failures),
            "warning_count":    len(warnings),
        },
        "warnings":  warnings,
        "failures":  failures,
    }

    # Print summary
    print(f"[A3] Chapters      : {len(chapter_backlogs)}")
    print(f"[A3] Total issues  : {len(all_issues)}")
    print(f"[A3] By severity   : {by_severity}")
    print(f"[A3] Health        : {by_health}")
    if warnings:
        print(f"[A3] Lint warnings : {len(warnings)}")
    if failures:
        print(f"[A3] FAILURES      : {len(failures)}", file=sys.stderr)
        for f in failures:
            print(f"  [{f['code']}] {f['message']}", file=sys.stderr)

    return packet


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print(
            "Usage: a3_repair_backlog_builder.py <input.json> <output.json>",
            file=sys.stderr,
        )
        sys.exit(1)

    input_path  = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    input_data = json.loads(input_path.read_text(encoding="utf-8"))
    packet     = run(input_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[A3] Output        : {output_path}")

    if packet["failures"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
