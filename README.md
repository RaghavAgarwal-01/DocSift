# LoRA Fine-Tuning: Receipt Field Extraction

Fine-tuned Qwen2.5-0.5B-Instruct with LoRA/PEFT to extract structured fields
(vendor, date, total) from raw OCR receipt text.

## Setup
- **Model:** Qwen/Qwen2.5-0.5B-Instruct
- **Method:** LoRA (r=8, alpha=16, target modules: q_proj, v_proj) via `peft`
- **Data:** `mychen76/invoices-and-receipts_ocr_v1` (Hugging Face) — 415 examples
  after filtering, OCR word lists paired with ground-truth structured fields
- **Training:** `trl` SFTTrainer, prompt/completion format (loss masked to the
  JSON answer only), 3 epochs, cosine LR schedule with warmup, LR 2e-5

## Results
| | Field-level exact-match accuracy |
|---|---|
| Baseline (zero-shot) | 12.22% |
| Fine-tuned (LoRA) | 14.44% |
| **Improvement** | **+2.22pp** |

## What I learned debugging this
This was a smaller improvement than I'd hoped for, and getting to a *correct*
measurement was most of the work:

- **Loss masking bug:** initially trained on a single merged prompt+completion
  string, which meant loss was computed over the entire noisy OCR input, not
  just the target JSON. Fixed by using TRL's `prompt`/`completion` column
  format, which enables completion-only loss automatically.
- **Attention/KV-cache bug:** the default SDPA attention implementation in
  this environment produced degenerate, repeating-token output after a few
  generated tokens — reproducible even zero-shot, unrelated to fine-tuning.
  Isolated via a bisection test (trivial prompt worked, task prompt didn't;
  short real prompt still broke) down to the attention implementation itself.
  Switching to `attn_implementation="eager"` fixed it.
- **Generation settings:** repetition penalty / no-repeat-ngram constraints,
  intended to prevent looping, actually hurt accuracy on this task, since
  receipts legitimately contain repeated tokens (e.g. "10%" tax lines).
  Removed them once the underlying attention bug was fixed.

## Why the accuracy gain is modest
- Small model (0.5B params), small training set (415 examples), 3 epochs —
  intentionally scoped small to be trainable on a free-tier GPU in a few
  hours
- Field-level exact-match is a strict metric — long address strings need to
  match character-for-character, so many "close" answers score as wrong
- With more time: a larger base model, more training data, and a softer
  metric (e.g. token F1 instead of exact match) would likely show a larger,
  more representative improvement

## Files
- `train.py` — full training + evaluation script
- `lora-receipt-extract-final/` — saved LoRA adapter weights
