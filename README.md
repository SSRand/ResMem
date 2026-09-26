# ResMem

Code for **From Lookups to Lessons: Turning Retrieved Knowledge into Parametric Memory**.

ResMem trains a feed-forward memory on top of a frozen Mistral-7B-v0.3 by residual self-distillation from a BM25-RAG teacher, and reads it out additively: softmax(z_B + γ s).

This anonymous version omits the implementation of a few data-processing components (teacher caching and training-batch assembly); they will be released upon publication.
Evaluation code is complete.

## Layout

| Directory | Contents | Paper |
| --- | --- | --- |
| `resmem/` | Memory architecture, readout rules, prompt, EM/F1 scoring, reference-answer NLL | all |
| `teacher/` | BM25 retrieval and teacher target caching | Sec. 3 |
| `train/` | ResMem trainer | Sec. 3 |
| `qa/` | Knowledge-QA data, inference, validation-set coefficient selection, aggregation (BM25-RAG under `qa/baselines/`) | Table 1 |
| `correction_harm/` | Correction/harm analysis | Figure 4 |
| `shared_teacher_ablation/` | Shared-teacher ablation | Table 2 |
| `precision/` | Precision and deployment cost | Appendix C |
| `gate/` | Tokenwise gate | Appendix D |

For the baselines (MLP Memory, Memory Decoder, LoRA, BM25-RAG) we provide evaluation code only.
Settings are recorded in [`docs/experimental_protocol.md`](docs/experimental_protocol.md).
Entry points are Python modules run from the repository root; each module's docstring gives its usage.

## Setup

```bash
pip install -r requirements.txt
```

## Knowledge QA (Table 1)

```bash
# Test and validation sets
python -m qa.data.build_test_sets --nq NQ_TEST --triviaqa TQA_TEST --webq WEBQ_TEST --hotpotqa HOTPOT_DEV --popqa POPQA --data-root data/qa/test
python -m qa.data.build_validation_sets --test-root data/qa/test/frozen --nq NQ_DEV --triviaqa TQA_DEV --webq WEBQ_DEV \
    --hotpotqa HOTPOT_TRAIN --popqa ENTITYQUESTIONS --output-root data/qa/validation

# Select the coefficient on validation EM, then evaluate on the test set
python -m qa.run_grid --split-root data/qa/validation/frozen --method resmem --checkpoint CKPT --coefficients GRID --output runs/resmem/validation
python -m qa.select_coefficients --split-root data/qa/validation/frozen --grid runs/resmem/validation --output runs/resmem/selected.json
python -m qa.run_grid --split-root data/qa/test/frozen --method resmem --checkpoint CKPT --selected runs/resmem/selected.json --score-answers --output runs/resmem/test
python -m qa.evaluate --split-root data/qa/test/frozen --method ResMem=runs/resmem/test:runs/resmem/selected.json
```

The same commands apply to `--method mlpmemory` and `--method memory_decoder`; `base` and `lora` take no coefficient.
