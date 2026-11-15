# Uncertainty-Representation-in-LLMs

## Purpose and Scope
- Investigates whether hidden representations inside DeepSeek-R1 models encode reliable signals about answer correctness on math problems.
- Focuses on samples from the HuggingFace `HuggingFaceH4/MATH-500` test split, collecting internal activations around reasoning markers to build a supervised probing dataset.
- Explores linear probes (logistic regression and linear SVM) as simple uncertainty estimators, along with geometric analysis of the learned decision boundary.

## Data Collection Pipeline
- `qwen1.5rollouts.ipynb` drives data generation using `nnsight`’s remote tracing API for the `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` model.
- For each math problem (looping up to 500 items) the notebook issues ~35 stochastic completions with `<think> … </think>` guidance and `max_new_tokens=2048`.
- During generation it records the final-layer hidden state for every token; sequences lacking a closing `</think>` token (ID `128014`) are discarded.
- A saved PyTorch bundle per problem (`rollouts/<problem_id>.pt`) contains:
  - full token batches,
  - captured hidden states aligned to the closing `</think>` position,
  - the prompt and solver instructions.
- Helper routines parse each rollout to pull the second `<think>` span, the post-think response, and the final boxed answer.

## Labeling and Dataset Assembly
- `probe.ipynb` rebuilds a consolidated `rollouts_token_df`:
  - downloads the MATH-500 metadata via the HF datasets server,
  - sanitizes problem IDs, attaches difficulty/subject labels, and maps gold answers,
  - decodes completions and extracts the last `\boxed{…}` expression with balanced-brace parsing.
- Final answers are checked against gold labels with `math_verify` to assign binary correctness targets.
- Problems with fewer than 20 usable attempts are dropped to ensure robust statistics.
- The notebook saves core probe inputs:
  - `X_4096.npy` – 4096-dim hidden vectors (one per attempt),
  - `y_labels.npy` – correctness labels,
  - `uids.npy` – stable identifiers.

## Probing and Evaluation
- `train_linear_probe` (within `probe.ipynb`) supports:
  - Scikit-learn logistic regression (SAGA) with L1/L2 search,
  - Skorch/PyTorch logistic regression with class-imbalance-aware loss,
  - Linear ThunderSVM with optional calibration for probabilistic outputs.
- Uses stratified 5-fold CV (scoring on F1) and a held-out test split; best Skorch configuration (lr≈3e-3, 60 epochs, weight decay 1e-3) reaches ≈0.89 test F1.
- Persists artifacts for downstream inspection:
  - `linear_probe.joblib`, `classification_plane.npz` (weights & bias in input space),
  - `projection_w_perp.npy`, `margin_direction_t.pt` (unit normal as a Torch tensor),
  - `cv_f1_table.csv`, `roc_pr_curves.png`, and `misclassified_examples.csv`.
- Additional cells quantify uncertainty with bootstrap confidence intervals and permutation tests, plus ROC/PR visualization.

## Representation Analysis
- Utility functions (`plot_margin_vs_pc1_perp`, `plot_margin_vs_pc2_perp`, `plot_margin_vs_pc12_3d_perp`) visualize margins against principal directions orthogonal to the learned plane.
- Overlays difficulty or subject metadata to inspect systematic structure; Spearman rank tests relate classifier margin/PC scores to task difficulty.
- `get_misclassified_examples` surfaces hardest failures with metadata and saved responses for qualitative review.

## Supporting Experiments
- `rollout.ipynb` contains earlier `nnsight` experiments (token tracing, minimal prompts) used to validate remote execution and hidden-state capture mechanics.
- `probe.ipynb` also sketches hooks into `nnsight` for layer-level interventions and cosine-similarity diagnostics relative to the learned margin direction.

## Dependencies
- Declared in `pyproject.toml`: `nnsight`, `datasets`, `math-verify`, `scikit-learn`, `skorch`, `thundersvm`, `einops`, `seaborn`, and core Jupyter tooling.
- Notebooks expect GPU access (for remote `nnsight` sessions or Skorch training) and outbound networking to reach the HF datasets server.
