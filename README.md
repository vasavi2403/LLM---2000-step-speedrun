# LLM---2000-step-speedrun
# Starter

Your loop:
  python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt
  python evaluate.py --checkpoint ckpt.pt --text_file ../data/dev_eval.txt

Baseline run ~1.5–3 min on a laptop CPU. Log every run in RUNLOG.md.
Everything in train.py and model.py is changeable; the caps and the
evaluate.py interface are not.

Before time is up, your submission folder needs: ckpt.pt, your code,
RUNLOG.md, NOTES.md, and SUMMARY.html (see the assignment brief,
"Deliverables" section).

# Training Run Log

**Run 1**
* **Hypothesis:** The baseline provided in the starter code will yield a poor score due to a static learning rate and no optimization.
* **What Changed:** Executed baseline `train.py` and `model.py` completely unmodified.
* **Dev BPB Before/After:** N/A -> 4.58
* **Conclusion:** The baseline is indeed mediocre on purpose. The model learns slowly and struggles with the English/Hindi corpus.

**Run 2**
* **Hypothesis:** The Hindi text (Devanagari) is bloating the sequence length because the standard byte-tokenizer uses up to 3 bytes per character. A character-level tokenizer with a byte-fallback will shrink sequence lengths and improve context utilization.
* **What Changed:** Replaced `tokenizer.py` with a custom `CharByteTokenizer` that maps all unique characters to a single token, falling back to raw bytes for unseen data.
* **Dev BPB Before/After:** 4.58 -> 3.25
* **Conclusion:** Massive improvement. Shrinking the token representations of Hindi characters allows the model to process much wider contextual windows within the same block size.

**Run 3**
* **Hypothesis:** The embedding table takes up too much of the 2,000,000 parameter budget. Tying the input embeddings to the output projection weights will free up parameters to make the network deeper.
* **What Changed:** Set `tie_weights = True` in `model.py`. Increased `n_layer` from 4 to 5, and `block_size` from 128 to 256. 
* **Dev BPB Before/After:** 3.25 -> 2.82
* **Conclusion:** Freeing up parameters allowed for an increase in network depth and sequence length without violating the 2M parameter cap.

**Run 4**
* **Hypothesis:** The static `3e-4` learning rate with the basic Adam optimizer limits convergence within the strict 2,000 step limit.
* **What Changed:** Implemented `AdamW` (weight decay = 0.1) and a Cosine Annealing Learning Rate Scheduler with a 200-step linear warmup, peaking at `3e-3` in `train.py`. Added gradient clipping (1.0).
* **Dev BPB Before/After:** 2.82 -> 2.15
* **Conclusion:** The aggressive early learning rate combined with the smooth cosine decay allowed the model to rapidly find a better minima within the exact 2,000 step constraint.

**Run 5**
* **Hypothesis:** Removing standard `LayerNorm` in favor of `RMSNorm` and tweaking the MLP layers will speed up CPU execution and provide a slight metric boost.
* **What Changed:** Swapped `nn.LayerNorm` for custom `RMSNorm` in `model.py`. Added `bias=False` to all Linear layers to shave off unnecessary parameters.
* **Dev BPB Before/After:** 2.15 -> 2.01
* **Conclusion:** The model executed faster on the CPU and parameters were utilized much more efficiently, pushing the BPB down further.

**Run 6 (Final)**
* **Hypothesis:** Pushing the batch size as high as the CPU can tolerate within a reasonable timeframe will yield the most stable gradient updates.
* **What Changed:** Increased `batch` from 8 to 16 in `train.py`.
* **Dev BPB Before/After:** 2.01 -> 1.84
* **Conclusion:** The larger batch size smoothed out the loss curve during the cosine decay phase, yielding the best final Bits-Per-Byte score.
