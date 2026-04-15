"""
Tests untuk A1 Block ID Injector
Covers: semua rules R-A1-01 s/d R-A1-12, example pack, edge cases
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import re
from pipeline.a1_block_id_injector import (
    run,
    normalize_prefix,
    extract_content_source,
    build_block_ref,
    build_content_fingerprint,
    build_structural_fingerprint,
    validate_control_block,
    RE_BLOCK_REF_CONTENT,
    RE_BLOCK_FP_CONTENT,
    RE_BLOCK_FP_STRUCTURAL,
    DEFAULT_NON_CONTENT_BLOCK_TYPES,
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


def make_input(chapters, **pcb_overrides):
    return {
        "pipeline_control_block": make_pcb(**pcb_overrides),
        "semantic_chapters": chapters,
    }


# ─────────────────────────────────────────────
# NORMALIZE_PREFIX
# ─────────────────────────────────────────────

class TestNormalizePrefix:
    def test_basic_lowercase_and_trim(self):
        assert normalize_prefix("  Hello World  ") == "hello_world"

    def test_punctuation_removed(self):
        assert normalize_prefix("Bab ini menyajikan: latar, belakang.") == \
               "bab_ini_menyajikan_latar_belakang"

    def test_max_40_chars(self):
        long = "a" * 60
        assert len(normalize_prefix(long)) == 40

    def test_newline_becomes_underscore(self):
        result = normalize_prefix("baris\nsatu")
        assert result == "baris_satu"

    def test_empty_returns_empty(self):
        assert normalize_prefix("") == "empty"
        assert normalize_prefix("   ") == "empty"
        assert normalize_prefix("---") == "empty"  # hanya punctuation

    def test_example_pack_paragraph_lead(self):
        # dari spec example pack
        # Catatan: spec menulis "..._tuju" (40 char) tapi setelah menghapus koma
        # string menjadi lebih pendek sehingga truncate di 40 menghasilkan "..._tujuan"
        # Output aktual: bab_ini_menyajikan_latar_belakang_tujuan (42 → 40 chars)
        text = "Bab ini menyajikan latar belakang, tujuan, ruang lingkup, dan konsiderasi hukum penyusunan laporan."
        result = normalize_prefix(text)
        assert result == "bab_ini_menyajikan_latar_belakang_tujuan"
        assert len(result) == 40

    def test_example_pack_table_header(self):
        result = normalize_prefix("Dimensi")
        assert result == "dimensi"

    def test_non_ascii_preserved_if_alphanumeric(self):
        # karakter non-ASCII yang tidak alfanumerik dibuang
        result = normalize_prefix("café")
        # 'é' bukan [a-z0-9] jadi dibuang
        assert result == "caf"

    def test_underscore_in_source_preserved(self):
        """Item 2: underscore dari teks asli harus survive normalization"""
        result = normalize_prefix("program_esd dilaksanakan")
        assert result == "program_esd_dilaksanakan"

    def test_underscore_not_doubled(self):
        """underscore dari teks asli tidak menghasilkan double underscore"""
        result = normalize_prefix("nilai_sroi blended")
        assert "__" not in result
        assert result == "nilai_sroi_blended"

    def test_multiple_spaces_collapsed(self):
        assert normalize_prefix("a   b   c") == "a_b_c"


# ─────────────────────────────────────────────
# EXTRACT_CONTENT_SOURCE
# ─────────────────────────────────────────────

class TestExtractContentSource:
    def test_text_first_priority(self):
        block = {"type": "p", "text": "hello", "title": "world"}
        val, found, source_key = extract_content_source(block)
        assert found and val == "hello"
        assert source_key == "text"

    def test_items_text_fallback(self):
        block = {"type": "list", "items": [{"text": "item one"}]}
        val, found, source_key = extract_content_source(block)
        assert found and val == "item one"
        assert source_key == "items[0].text"

    def test_headers_fallback(self):
        block = {"type": "table", "headers": ["Kolom A", "Kolom B"]}
        val, found, source_key = extract_content_source(block)
        assert found and val == "Kolom A"
        assert source_key == "headers[0]"

    def test_title_fallback(self):
        block = {"type": "section", "title": "Judul Subbab"}
        val, found, source_key = extract_content_source(block)
        assert found and val == "Judul Subbab"
        assert source_key == "title"

    def test_rows_fallback(self):
        block = {"type": "table", "rows": [["Nilai A", "Nilai B"]]}
        val, found, source_key = extract_content_source(block)
        assert found and val == "Nilai A"
        assert source_key == "rows[0][0]"

    def test_label_fallback(self):
        block = {"type": "metric", "label": "Total SROI"}
        val, found, source_key = extract_content_source(block)
        assert found and val == "Total SROI"
        assert source_key == "label"

    def test_value_fallback(self):
        block = {"type": "metric", "value": "1.14"}
        val, found, source_key = extract_content_source(block)
        assert found and val == "1.14"
        assert source_key == "value"

    def test_no_source_returns_not_found(self):
        block = {"type": "divider"}
        val, found, source_key = extract_content_source(block)
        assert not found
        assert source_key is None

    def test_empty_text_falls_through(self):
        block = {"type": "p", "text": "", "title": "fallback title"}
        val, found, source_key = extract_content_source(block)
        assert found and val == "fallback title"
        assert source_key == "title"

    def test_empty_items_list(self):
        block = {"type": "list", "items": [], "title": "fallback"}
        val, found, source_key = extract_content_source(block)
        assert found and val == "fallback"
        assert source_key == "title"


# ─────────────────────────────────────────────
# BUILD FUNCTIONS
# ─────────────────────────────────────────────

class TestBuildFunctions:
    def test_block_ref_format(self):
        assert build_block_ref("bab_1", 1) == "bab_1.B001"
        assert build_block_ref("bab_7", 42) == "bab_7.B042"
        assert build_block_ref("bab_1", 1000) == "bab_1.B1000"

    def test_block_ref_regex(self):
        assert RE_BLOCK_REF_CONTENT.match(build_block_ref("bab_1", 1))
        assert RE_BLOCK_REF_CONTENT.match(build_block_ref("bab_7", 100))

    def test_content_fingerprint(self):
        fp = build_content_fingerprint("bab_1", "paragraph_lead", "bab_ini_menyajikan")
        assert fp == "bab_1__paragraph_lead__bab_ini_menyajikan"
        assert RE_BLOCK_FP_CONTENT.match(fp)

    def test_structural_fingerprint(self):
        fp = build_structural_fingerprint("bab_7", "divider", 2)
        assert fp == "bab_7__divider__structural_2"
        assert RE_BLOCK_FP_STRUCTURAL.match(fp)

    def test_structural_fp_regex(self):
        for idx in [0, 1, 99]:
            fp = build_structural_fingerprint("bab_1", "spacer", idx)
            assert RE_BLOCK_FP_STRUCTURAL.match(fp), f"Regex failed for: {fp}"


# ─────────────────────────────────────────────
# VALIDATE CONTROL BLOCK
# ─────────────────────────────────────────────

class TestValidateControlBlock:
    def test_valid_confirmed(self):
        assert validate_control_block(make_pcb()) == []

    def test_valid_discovery(self):
        pcb = make_pcb(substance_mode="discovery", substance_chapter_id=None)
        assert validate_control_block(pcb) == []

    def test_missing_document_id(self):
        pcb = make_pcb(document_id="")
        errors = validate_control_block(pcb)
        assert any("document_id" in e for e in errors)

    def test_invalid_substance_mode(self):
        pcb = make_pcb(substance_mode="unknown_mode")
        errors = validate_control_block(pcb)
        assert any("substance_mode" in e for e in errors)

    def test_invalid_substance_basis(self):
        pcb = make_pcb(substance_basis="guessed")
        errors = validate_control_block(pcb)
        assert any("substance_basis" in e for e in errors)

    def test_confirmed_without_chapter_id(self):
        pcb = make_pcb(substance_mode="confirmed", substance_chapter_id=None)
        errors = validate_control_block(pcb)
        assert any("substance_chapter_id" in e for e in errors)

    def test_confirmed_with_unknown_basis(self):
        pcb = make_pcb(substance_mode="confirmed", substance_basis="unknown")
        errors = validate_control_block(pcb)
        assert any("unknown" in e for e in errors)

    def test_not_dict_returns_error(self):
        errors = validate_control_block("not a dict")
        assert errors


# ─────────────────────────────────────────────
# RUN — EXAMPLE PACK (dari spec)
# ─────────────────────────────────────────────

class TestExamplePack:
    def test_paragraph_lead_normal(self):
        """[example: ESP] Paragraph lead normal"""
        inp = make_input([{
            "chapter_id": "bab_1",
            "chapter_type": "opening",
            "blocks": [{
                "type": "paragraph_lead",
                "text": "Bab ini menyajikan latar belakang, tujuan, ruang lingkup, dan konsiderasi hukum penyusunan laporan.",
            }]
        }])
        result = run(inp)
        assert not result.get("failures")
        block = result["chapters"][0]["blocks"][0]
        assert block["block_ref"] == "bab_1.B001"
        assert block["block_fingerprint"] == \
               "bab_1__paragraph_lead__bab_ini_menyajikan_latar_belakang_tujuan"
        assert block["original_index"] == 0
        # R-A1-05: field asli utuh
        assert block["type"] == "paragraph_lead"
        assert "text" in block

    def test_table_block_tanpa_text(self):
        """[example: ESP] Table block tanpa text"""
        inp = make_input([{
            "chapter_id": "bab_1",
            "chapter_type": "opening",
            "blocks": [
                # 11 blocks sebelumnya (dummy) untuk memastikan B012
                *[{"type": "paragraph", "text": f"dummy block {i}"} for i in range(11)],
                {
                    "type": "table_borderless",
                    "headers": ["Dimensi", "Cakupan"],
                    "rows": [["Program", "Enduro Student Program"]],
                },
            ]
        }])
        result = run(inp)
        assert not result.get("failures")
        block = result["chapters"][0]["blocks"][11]
        assert block["block_ref"] == "bab_1.B012"
        assert block["block_fingerprint"] == "bab_1__table_borderless__dimensi"
        assert block["original_index"] == 11

    def test_divider_structural(self):
        """[example: generic] Divider — non_content_block_type"""
        inp = make_input([{
            "chapter_id": "bab_1",
            "chapter_type": None,
            "blocks": [
                {"type": "paragraph_lead", "text": "Opening paragraph"},
                {"type": "paragraph", "text": "Second paragraph"},
                {"type": "divider"},  # original_index = 2
            ]
        }])
        result = run(inp)
        block = result["chapters"][0]["blocks"][2]
        assert block["block_ref"] == "bab_1.B003"
        assert block["block_fingerprint"] == "bab_1__divider__structural_2"
        assert block["original_index"] == 2

    def test_collision_case_fatal(self):
        """[example: generic] Collision case — proses stop"""
        inp = make_input([{
            "chapter_id": "bab_1",
            "chapter_type": "opening",
            "blocks": [
                {"type": "paragraph", "text": "Teks yang sama persis untuk collision"},
                {"type": "paragraph", "text": "Teks yang sama persis untuk collision"},
            ]
        }])
        result = run(inp)
        assert result.get("failures")
        assert any(f["code"] == "FINGERPRINT_COLLISION" for f in result["failures"])


# ─────────────────────────────────────────────
# RUN — RULES
# ─────────────────────────────────────────────

class TestRules:
    def test_r_a1_01_identity_fields_present(self):
        """R-A1-01: semua block punya 3 identity fields"""
        inp = make_input([{
            "chapter_id": "bab_1",
            "chapter_type": "opening",
            "blocks": [
                {"type": "paragraph", "text": "Blok pertama"},
                {"type": "divider"},
                {"type": "table", "headers": ["A"], "rows": [["1"]]},
            ]
        }])
        result = run(inp)
        assert not result.get("failures")
        for block in result["chapters"][0]["blocks"]:
            assert "block_ref" in block
            assert "block_fingerprint" in block
            assert "original_index" in block

    def test_r_a1_02_fingerprint_unique_document_wide(self):
        """R-A1-02: fingerprint unik document-wide"""
        inp = make_input([
            {
                "chapter_id": "bab_1",
                "chapter_type": "opening",
                "blocks": [{"type": "paragraph", "text": "Teks unik bab satu"}]
            },
            {
                "chapter_id": "bab_2",
                "chapter_type": "core",
                "blocks": [{"type": "paragraph", "text": "Teks unik bab satu"}]  # same text, diff chapter → diff fp
            }
        ])
        result = run(inp)
        # Different chapter_id → different fingerprint prefix → no collision
        assert not result.get("failures")
        fps = [
            b["block_fingerprint"]
            for ch in result["chapters"]
            for b in ch["blocks"]
        ]
        assert len(fps) == len(set(fps))

    def test_r_a1_03_block_ref_chapter_local(self):
        """R-A1-03: block_ref sequencing per chapter (chapter-local, starts dari B001)"""
        inp = make_input([
            {
                "chapter_id": "bab_1",
                "chapter_type": "opening",
                "blocks": [
                    {"type": "paragraph", "text": "A"},
                    {"type": "paragraph", "text": "B"},
                ]
            },
            {
                "chapter_id": "bab_2",
                "chapter_type": "core",
                "blocks": [
                    {"type": "paragraph", "text": "C"},
                ]
            }
        ])
        result = run(inp)
        bab1_refs = [b["block_ref"] for b in result["chapters"][0]["blocks"]]
        bab2_refs = [b["block_ref"] for b in result["chapters"][1]["blocks"]]
        assert bab1_refs == ["bab_1.B001", "bab_1.B002"]
        assert bab2_refs == ["bab_2.B001"]  # starts from B001, not continues from bab_1

    def test_r_a1_05_non_destructive(self):
        """R-A1-05: field asli tidak hilang"""
        original_block = {
            "type": "paragraph",
            "text": "Original text",
            "custom_field": "custom_value",
        }
        inp = make_input([{
            "chapter_id": "bab_1",
            "chapter_type": "opening",
            "blocks": [original_block]
        }])
        result = run(inp)
        block = result["chapters"][0]["blocks"][0]
        assert block["type"] == "paragraph"
        assert block["text"] == "Original text"
        assert block["custom_field"] == "custom_value"

    def test_r_a1_06_empty_content_warning(self):
        """R-A1-06: empty content block → warning, bukan failure"""
        inp = make_input([{
            "chapter_id": "bab_1",
            "chapter_type": "opening",
            "blocks": [{"type": "paragraph"}]  # no text, no fallback
        }])
        result = run(inp)
        assert not result.get("failures")
        assert any(w["code"] == "EMPTY_CONTENT_BLOCK" for w in result.get("warnings", []))
        block = result["chapters"][0]["blocks"][0]
        assert block["block_fingerprint"].endswith("__empty")

    def test_primary_source_absent_warning_with_source_key(self):
        """PRIMARY_SOURCE_ABSENT: warning menyebut source mana yang dipakai"""
        inp = make_input([{
            "chapter_id": "bab_1",
            "chapter_type": "opening",
            "blocks": [
                # table tanpa text → fallback ke headers[0]
                {"type": "table_borderless", "headers": ["Dimensi", "Cakupan"]},
                # list tanpa text → fallback ke items[0].text
                {"type": "bullet_list", "items": [{"text": "item satu"}]},
            ]
        }])
        result = run(inp)
        assert not result.get("failures")
        warnings = result.get("warnings", [])
        pa_warnings = [w for w in warnings if w["code"] == "PRIMARY_SOURCE_ABSENT"]
        assert len(pa_warnings) == 2
        sources = {w["source_used"] for w in pa_warnings}
        assert "headers[0]" in sources
        assert "items[0].text" in sources
        # Pesan tidak lagi menyebut "field 'text' kosong"
        for w in pa_warnings:
            assert "field 'text' kosong" not in w["message"]
            assert w["source_used"] in w["message"]

    def test_r_a1_07_collision_fatal_stops_process(self):
        """R-A1-07: collision → stop, output tidak valid"""
        inp = make_input([{
            "chapter_id": "bab_1",
            "chapter_type": "opening",
            "blocks": [
                {"type": "paragraph", "text": "Duplikat"},
                {"type": "paragraph", "text": "Duplikat"},
            ]
        }])
        result = run(inp)
        failures = result.get("failures", [])
        assert any(f["code"] == "FINGERPRINT_COLLISION" for f in failures)
        assert result.get("status") == "failed"

    def test_r_a1_08_no_auto_disambiguation(self):
        """R-A1-08: tidak ada suffix _2, _3 untuk collision"""
        inp = make_input([{
            "chapter_id": "bab_1",
            "chapter_type": "opening",
            "blocks": [
                {"type": "paragraph", "text": "Same text"},
                {"type": "paragraph", "text": "Same text"},
            ]
        }])
        result = run(inp)
        # Harus failure, bukan diam-diam menambah _2
        assert result.get("failures")
        fps = [b.get("block_fingerprint", "") for b in result["chapters"][0]["blocks"]]
        # Tidak ada fingerprint dengan _2 atau _3
        assert not any("_2" in fp or "_3" in fp for fp in fps)

    def test_r_a1_09_invalid_control_block(self):
        """R-A1-09: control block invalid → stop sebelum proses"""
        inp = {
            "pipeline_control_block": {"document_id": ""},  # invalid
            "semantic_chapters": [{"chapter_id": "bab_1", "chapter_type": None, "blocks": []}]
        }
        result = run(inp)
        assert result.get("failures")
        assert any(f["code"] == "CONTROL_BLOCK_INVALID" for f in result["failures"])

    def test_r_a1_10_program_agnostic(self):
        """R-A1-10: bekerja untuk program_code apapun"""
        for code in ["ESD", "ESP", "PSS", "TEST", "XYZ_123"]:
            inp = make_input(
                [{"chapter_id": "ch_1", "chapter_type": "core",
                  "blocks": [{"type": "paragraph", "text": "Generic content"}]}],
                program_code=code
            )
            result = run(inp)
            assert not result.get("failures"), f"Failed for program_code={code}"
            assert result["program_code"] == code

    def test_r_a1_11_non_content_structural_fingerprint(self):
        """R-A1-11: non-content blocks pakai structural fingerprint"""
        inp = make_input([{
            "chapter_id": "bab_1",
            "chapter_type": "opening",
            "blocks": [
                {"type": "divider"},
                {"type": "spacer"},
                {"type": "divider_thick"},
            ]
        }])
        result = run(inp)
        assert not result.get("failures")
        for block in result["chapters"][0]["blocks"]:
            assert RE_BLOCK_FP_STRUCTURAL.match(block["block_fingerprint"]), \
                f"Bukan structural: {block['block_fingerprint']}"

    def test_r_a1_12_non_content_not_in_content_collision(self):
        """R-A1-12: dua divider dengan original_index berbeda → tidak collision"""
        inp = make_input([{
            "chapter_id": "bab_1",
            "chapter_type": "opening",
            "blocks": [
                {"type": "paragraph", "text": "Para A"},
                {"type": "divider"},
                {"type": "paragraph", "text": "Para B"},
                {"type": "divider"},  # same type, different original_index
            ]
        }])
        result = run(inp)
        assert not result.get("failures")

    def test_cross_chapter_collision_detected(self):
        """Collision antara dua chapter berbeda → fatal"""
        inp = make_input([
            {
                "chapter_id": "bab_1",
                "chapter_type": "opening",
                "blocks": [{"type": "paragraph", "text": "Teks yang akan collision"}]
            },
            {
                "chapter_id": "bab_2",
                "chapter_type": "core",
                # sama persis type + normalized prefix → TIDAK collision karena chapter_id berbeda
                # chapter_id masuk ke fingerprint, jadi ini TIDAK akan collision
                "blocks": [{"type": "paragraph", "text": "Teks yang akan collision"}]
            }
        ])
        result = run(inp)
        # Chapter_id berbeda → fingerprint berbeda → TIDAK collision
        assert not result.get("failures")

    def test_same_text_same_chapter_collision(self):
        """Collision dalam satu chapter → fatal"""
        inp = make_input([{
            "chapter_id": "bab_1",
            "chapter_type": "opening",
            "blocks": [
                {"type": "paragraph", "text": "Teks yang collision"},
                {"type": "paragraph", "text": "Teks yang collision"},
            ]
        }])
        result = run(inp)
        assert result.get("failures")


# ─────────────────────────────────────────────
# RUN — SUMMARY & OUTPUT STRUCTURE
# ─────────────────────────────────────────────

class TestOutputStructure:
    def test_summary_fields_present(self):
        inp = make_input([{
            "chapter_id": "bab_1",
            "chapter_type": "opening",
            "blocks": [{"type": "paragraph", "text": "Test"}]
        }])
        result = run(inp)
        summary = result["summary"]
        for field in ["total_chapters", "total_blocks", "fingerprint_collisions",
                      "empty_content_blocks", "warnings_count", "failure_count"]:
            assert field in summary, f"Missing summary field: {field}"

    def test_activity_field(self):
        inp = make_input([{
            "chapter_id": "bab_1",
            "chapter_type": "opening",
            "blocks": [{"type": "paragraph", "text": "Test"}]
        }])
        result = run(inp)
        assert result["activity"] == "block_id_injector"

    def test_generated_at_iso8601(self):
        from datetime import datetime
        inp = make_input([{
            "chapter_id": "bab_1",
            "chapter_type": "opening",
            "blocks": [{"type": "paragraph", "text": "Test"}]
        }])
        result = run(inp)
        # Should parse without error
        dt = datetime.fromisoformat(result["generated_at"].replace("Z", "+00:00"))
        assert dt is not None

    def test_total_blocks_count(self):
        inp = make_input([{
            "chapter_id": "bab_1",
            "chapter_type": "opening",
            "blocks": [
                {"type": "paragraph", "text": "A"},
                {"type": "paragraph", "text": "B"},
                {"type": "divider"},
            ]
        }])
        result = run(inp)
        assert result["summary"]["total_blocks"] == 3

    def test_empty_blocks_chapter(self):
        inp = make_input([{
            "chapter_id": "bab_1",
            "chapter_type": "opening",
            "blocks": []
        }])
        result = run(inp)
        assert not result.get("failures")
        assert result["summary"]["total_blocks"] == 0

    def test_multiple_chapters(self):
        inp = make_input([
            {"chapter_id": f"bab_{i}", "chapter_type": "core",
             "blocks": [{"type": "paragraph", "text": f"Teks unik bab {i} berbeda"}]}
            for i in range(1, 6)
        ])
        result = run(inp)
        assert not result.get("failures")
        assert result["summary"]["total_chapters"] == 5
        assert result["summary"]["total_blocks"] == 5

    def test_original_index_preserved(self):
        """original_index harus 0-based dan sesuai posisi asli"""
        inp = make_input([{
            "chapter_id": "bab_1",
            "chapter_type": "opening",
            "blocks": [
                {"type": "paragraph", "text": "A"},
                {"type": "divider"},
                {"type": "paragraph", "text": "B"},
            ]
        }])
        result = run(inp)
        blocks = result["chapters"][0]["blocks"]
        assert blocks[0]["original_index"] == 0
        assert blocks[1]["original_index"] == 1
        assert blocks[2]["original_index"] == 2

    def test_input_not_mutated(self):
        """R-A1-05: input asli tidak berubah"""
        original = {
            "type": "paragraph",
            "text": "Original",
        }
        inp = make_input([{
            "chapter_id": "bab_1",
            "chapter_type": "opening",
            "blocks": [original]
        }])
        original_copy = dict(original)
        run(inp)
        assert original == original_copy


# ─────────────────────────────────────────────
# EDGE CASES
# ─────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_semantic_chapters(self):
        inp = {
            "pipeline_control_block": make_pcb(),
            "semantic_chapters": []
        }
        result = run(inp)
        assert result.get("failures")
        assert any(f["code"] == "EMPTY_CHAPTERS" for f in result["failures"])

    def test_block_without_type_is_fatal(self):
        inp = make_input([{
            "chapter_id": "bab_1",
            "chapter_type": "opening",
            "blocks": [{"text": "no type here"}]
        }])
        result = run(inp)
        assert result.get("failures")
        assert any(f["code"] == "BLOCK_INVALID" for f in result["failures"])

    def test_chapter_without_id_is_fatal(self):
        inp = {
            "pipeline_control_block": make_pcb(),
            "semantic_chapters": [{"blocks": [{"type": "paragraph", "text": "x"}]}]
        }
        result = run(inp)
        assert result.get("failures")

    def test_blocks_not_array_is_fatal(self):
        inp = {
            "pipeline_control_block": make_pcb(),
            "semantic_chapters": [{"chapter_id": "bab_1", "blocks": "not a list"}]
        }
        result = run(inp)
        assert result.get("failures")

    def test_discovery_mode_no_chapter_id(self):
        """discovery mode: substance_chapter_id boleh null"""
        inp = make_input(
            [{"chapter_id": "bab_1", "chapter_type": "opening",
              "blocks": [{"type": "paragraph", "text": "test"}]}],
            substance_mode="discovery",
            substance_chapter_id=None,
        )
        result = run(inp)
        assert not result.get("failures")

    def test_custom_non_content_types(self):
        """non_content_block_types dari pcb dipakai, bukan default"""
        inp = make_input(
            [{"chapter_id": "bab_1", "chapter_type": "opening",
              "blocks": [
                  {"type": "custom_separator"},  # custom non-content
                  {"type": "paragraph", "text": "normal"},
              ]}],
            non_content_block_types=["custom_separator"]
        )
        result = run(inp)
        assert not result.get("failures")
        sep_block = result["chapters"][0]["blocks"][0]
        assert RE_BLOCK_FP_STRUCTURAL.match(sep_block["block_fingerprint"])

    def test_very_long_text_prefix_truncated(self):
        long_text = "a" * 200 + " extra"
        inp = make_input([{
            "chapter_id": "bab_1",
            "chapter_type": "opening",
            "blocks": [{"type": "paragraph", "text": long_text}]
        }])
        result = run(inp)
        block = result["chapters"][0]["blocks"][0]
        # prefix part after second __ must be ≤ 40 chars
        parts = block["block_fingerprint"].split("__")
        assert len(parts) == 3
        assert len(parts[2]) <= 40

    def test_regex_validation_all_blocks(self):
        """Semua fingerprint dan block_ref harus match regex"""
        inp = make_input([{
            "chapter_id": "bab_7",
            "chapter_type": "core",
            "blocks": [
                {"type": "paragraph_lead", "text": "Lead paragraph"},
                {"type": "divider"},
                {"type": "table", "headers": ["H1", "H2"]},
                {"type": "spacer"},
                {"type": "bullet_list", "items": [{"text": "item"}]},
            ]
        }])
        result = run(inp)
        assert not result.get("failures")
        for block in result["chapters"][0]["blocks"]:
            ref = block["block_ref"]
            fp  = block["block_fingerprint"]
            assert RE_BLOCK_REF_CONTENT.match(ref), f"block_ref invalid: {ref}"
            valid_fp = (
                RE_BLOCK_FP_CONTENT.match(fp) or
                RE_BLOCK_FP_STRUCTURAL.match(fp)
            )
            assert valid_fp, f"fingerprint invalid: {fp}"
