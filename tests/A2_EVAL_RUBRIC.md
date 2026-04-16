# A2 Substance Extractor — Eval Rubric
# Golden Sample Review (terpisah dari test suite teknis)
# Dipakai untuk review manual 3–5 output A2 setelah prompt patch

## Cara pakai

Setelah A2 dijalankan, bandingkan output `substance_registry.json`
dengan rubrik ini. Bukan checklist binary — ini adalah panduan penilaian editorial.

---

## Dimensi 1 — Abstraksi vs Extraction

**Yang diharapkan:**
Elemen harus bisa berdiri sendiri di luar Bab 7. Summary harus bisa
dimengerti oleh penulis Bab 1 atau Bab 9 tanpa membaca Bab 7.

**Sinyal output yang baik:**
- Summary mengandung konteks epistemik ("berdasarkan transaksi aktual",
  "diestimasi menggunakan proxy", "menunggu verifikasi dokumen keuangan")
- Label mencerminkan substansi, bukan heading Bab 7

**Sinyal output yang buruk:**
- Summary hanya mengulang angka tanpa konteks
- Label seperti "section_74_investasi" atau "bagian_outcome"
- Satu block → satu elemen tanpa pertimbangan grouping

**Golden reference — elemen investasi (target):**
```json
{
  "label": "investment_basis_and_confirmation_status",
  "summary": "Basis investasi program ESD selama 2023–2025 berjumlah Rp 594,781,486,
    tetapi struktur evidensinya tidak seragam. Nilai total kumulatif dapat diperlakukan
    sebagai anchor evaluatif, sementara komponen tahunan 2023–2024 masih perlu dibaca
    sebagai angka operasional yang menunggu verifikasi dari dokumen keuangan resmi.",
  "source_block_types": ["callout_warning", "paragraph", "table", "metric_card_3col"],
  "evidence_status": "mixed"
}
```

---

## Dimensi 2 — Grouping

**Yang diharapkan:**
Blocks yang membentuk satu unit makna harus digabung.
Blocks dengan level evidence berbeda harus dipisah.

**Sinyal output yang baik:**
- Elemen investasi menggabungkan callout_warning + paragraph + table
- Outcome observed dan outcome proxy menjadi dua elemen terpisah
- Elemen SROI metric menggabungkan paragraph_lead + metric_card + tabel kalkulasi + tabel ringkasan

**Sinyal output yang buruk:**
- Setiap block menjadi satu elemen
- Observed dan proxy digabung menjadi satu elemen "program_outcomes"

**Golden reference — dua elemen outcome (target):**
```
elemen 1: observed_outcome_market_activation
  evidence_status: final
  source: blocks observed transaction

elemen 2: proxy_outcome_activation_independence
  evidence_status: proxy
  source: blocks proxy (kapasitas usaha, kemandirian, self-efficacy, pengakuan sosial)
```

---

## Dimensi 3 — use_affordances

**Yang diharapkan:**
Elemen material harus punya minimal 3 affordances.
Affordance harus mencerminkan fungsi lintas bab, bukan hanya bab sumber.

**Sinyal output yang baik:**
- SUB untuk SROI metric punya: opening_mandate, methodological_reference,
  outcome_reference, adjustment_reference, monetization_reference,
  learning_anchor, closing_summary, recommendation_basis, visual_candidate
- Elemen investasi punya: opening_mandate, scope_definition,
  implementation_reference, methodological_reference, closing_summary

**Sinyal output yang buruk:**
- Mayoritas elemen hanya punya 1 affordance
- Affordance hanya mencerminkan Bab 7 (misalnya hanya implementation_reference)

**Threshold minimum yang diterima:**
- Elemen dengan materiality >= 4: minimal 3 affordances
- Elemen dengan materiality 3: minimal 2 affordances
- Elemen dengan materiality <= 2: minimal 1 affordance

---

## Dimensi 4 — evidence_status

**Yang diharapkan:**
Tidak semua elemen boleh "final". Pembedaan harus sadar dan defensible.

**Decision rules:**
- final: transaksi aktual, dokumen keuangan terverifikasi
- proxy: estimasi dari referensi kebijakan atau benchmark
- mixed: campuran final + proxy/inferred dalam satu elemen
- inferred: ditarik dari implikasi logis, tanpa data langsung
- pending: data ada tapi belum diverifikasi

**Sinyal output yang buruk:**
- Semua elemen = "final" → trigger LINT_ALL_EVIDENCE_FINAL
- Outcome proxy diberi evidence_status "final"
- Investasi 2023–2024 (under confirmation) diberi "final"

**Golden reference untuk ESD:**
- investment_basis → mixed (ada under_confirmation)
- observed_outcome → final
- proxy_outcome → proxy
- sroi_metric → final (kalkulasi deterministik dari data terverifikasi)

---

## Dimensi 5 — guardrail_notes

**Yang diharapkan:**
Setiap elemen material (materiality >= 4) harus punya minimal 1 guardrail_note.
Format: larangan atau syarat, bukan aspirasi.

**Format yang diterima:**
- "Jangan tampilkan angka X tanpa menyebut komposisi observed vs proxy."
- "Jika dibawa ke Bab I, perlakukan sebagai mandat evaluatif, bukan kesimpulan."
- "Jangan gabungkan elemen ini dengan proxy outcome tanpa membedakan level evidensinya."
- "Hanya boleh dikutip setelah metodologi DDAT dijelaskan di Bab III."

**Format yang ditolak:**
- "Ensure accuracy of data."
- "Verify alignment with program objectives."
- "Make sure this is transparent."

**Sinyal output yang buruk:**
- guardrail_notes = [] untuk elemen dengan materiality >= 4
  → trigger LINT_HIGH_MATERIALITY_NO_GUARDRAIL
- Semua guardrail_notes berformat "Ensure/Verify/Make sure"
  → trigger LINT_ASPIRATIONAL_GUARDRAIL

---

## Scoring Ringkasan

Setelah review manual, nilai output dengan skala berikut:

| Skor | Arti |
|---|---|
| 5 | Output abstraktif, grouping tepat, affordances lengkap, evidence jujur, guardrails operasional |
| 4 | Sebagian besar benar, 1-2 elemen masih terlalu atomik atau affordance kurang |
| 3 | Lolos teknis, tapi masih dominan extractive bukan abstractive |
| 2 | Banyak elemen atomik, guardrail kosong atau generik, evidence semua final |
| 1 | Output tidak lebih baik dari sebelum patch prompt |

**Target minimum setelah patch: skor 4.**
Jika masih 3 atau di bawah, audit prompt kembali sebelum lanjut ke Sprint 3.

---

## Semantic Lint Codes — Referensi

| Code | Artinya |
|---|---|
| LINT_ALL_EVIDENCE_FINAL | Semua elemen evidence=final, curigai over-confidence |
| LINT_ALL_GUARDRAIL_NOTES_EMPTY | Pass 1 dan Pass 2 tidak mengisi guardrail_notes |
| LINT_AFFORDANCES_UNDERSPECIFIED | Mayoritas elemen hanya 1 affordance |
| LINT_HIGH_MATERIALITY_NO_GUARDRAIL | Elemen penting tanpa guardrail |
| LINT_ASPIRATIONAL_GUARDRAIL | Guardrail berbunyi "Ensure/Verify" — tidak operasional |
