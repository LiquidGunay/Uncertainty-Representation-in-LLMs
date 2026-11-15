#!/usr/bin/env python3
"""Build probe-ready NumPy arrays from rollout `.json.pt` bundles.

This utility mirrors the data assembly logic in `probe.ipynb`:

* loads every rollout saved by `qwen1.5rollouts.ipynb` (per-problem `.json.pt`),
* extracts the hidden-state vectors captured at the close of the thinking block,
* decodes generations to pull the final `\boxed{...}` answer,
* compares against the MATH-500 gold answer (optionally via `math_verify`),
* and saves aligned `X_4096.npy`, `y_labels.npy`, and optional `uids.npy` files.

Attempts lacking a boxed answer, or problems without a gold solution, are
discarded. By default, problems with fewer than 20 usable attempts are also
pruned to stay consistent with the notebook; override via `--min-attempts`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _lazy_import(module_name: str):
    try:
        module = __import__(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive
        raise SystemExit(
            f"Missing dependency '{module_name}'. Install it before running this script."
        ) from exc
    return module


def _lazy_import_torch():
    return _lazy_import("torch")


def _lazy_tokenizer(spec: str):
    transformers = _lazy_import("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(  # type: ignore[attr-defined]
            spec,
            trust_remote_code=True,
        )
    except Exception as exc:  # pragma: no cover - download/config issues
        raise SystemExit(f"Failed to load tokenizer '{spec}': {exc}") from exc
    return tokenizer


def _maybe_load_math_verify(use_math_verify: bool):
    if not use_math_verify:
        return None
    try:
        from math_verify import parse, verify  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        print(
            "[convert-rollouts] math_verify unavailable, falling back to string comparison",
            file=sys.stderr,
        )
        return None
    return parse, verify


def _as_2d_tensor(tensor_like, torch_module, expected_dim: int):
    torch = torch_module
    if isinstance(tensor_like, torch.Tensor):
        tensor = tensor_like
    elif isinstance(tensor_like, (list, tuple)) and tensor_like:
        tensor = torch.stack([torch.as_tensor(x) for x in tensor_like])
    else:
        raise TypeError("hidden_states must be a Tensor or non-empty sequence of Tensors")

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 2:
        raise ValueError(f"Expected hidden_states with 1 or 2 dims, got {tensor.ndim}")

    hidden_dim = tensor.shape[-1]
    if expected_dim and hidden_dim != expected_dim:
        raise ValueError(
            f"Hidden state width {hidden_dim} does not match expected {expected_dim}"
        )

    return tensor, hidden_dim


def _iter_rollout_files(root: Path) -> Iterable[Path]:
    return sorted(root.glob("*.json.pt"))


def _coerce_prompt_ids(prompt, torch_module) -> List[int]:
    torch = torch_module
    if prompt is None:
        return []
    if isinstance(prompt, torch.Tensor):
        return prompt.detach().cpu().tolist()
    if isinstance(prompt, (list, tuple)):
        return [int(x) for x in prompt]
    if isinstance(prompt, dict) and "input_ids" in prompt:
        return [int(x) for x in prompt["input_ids"]]
    raise TypeError(f"Unsupported prompt type: {type(prompt)!r}")


def _extract_last_boxed(text: str) -> str:
    matches = list(re.finditer(r"\\boxed\s*\{", text))
    if not matches:
        return ""
    start_brace_idx = matches[-1].end() - 1
    start = start_brace_idx + 1
    depth = 1
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].replace(" ", "").rstrip("\\%")
        i += 1
    return text[start:]


def _load_math_metadata(metadata_path: Optional[Path]) -> Tuple[dict, dict]:
    if metadata_path is not None:
        payload = json.loads(metadata_path.read_text())
        if isinstance(payload, dict) and "rows" in payload:
            rows = [row.get("row", row) for row in payload["rows"]]
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError("Unsupported metadata JSON structure")
    else:
        requests = _lazy_import("requests")
        rows = []
        for offset in range(0, 500, 100):
            url = (
                "https://datasets-server.huggingface.co/rows"
                "?dataset=HuggingFaceH4/MATH-500&config=default&split=test"
                f"&offset={offset}&length=100"
            )
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            json_rows = resp.json()["rows"]
            rows.extend(row["row"] for row in json_rows)

    ans_map = {
        row["unique_id"].replace("/", ""): row["answer"]
        for row in rows
        if "answer" in row
    }
    level_map = {
        row["unique_id"].replace("/", ""): row.get("level")
        for row in rows
    }
    return ans_map, level_map


def _answer_equivalent(pred: str, gold: str, math_verify_module) -> bool:
    if not pred or not gold:
        return False

    pred_clean = pred.rstrip("\\%")
    gold_clean = gold.rstrip("\\%")

    if math_verify_module is not None:
        parse, verify = math_verify_module
        try:
            lhs = parse(r"\\boxed{" + gold_clean + "}")
            rhs = parse(r"\\boxed{" + pred_clean + "}")
            return bool(verify(lhs, rhs))
        except Exception:
            pass

    def _normalize(text: str) -> str:
        return re.sub(r"\s+", "", text)

    return _normalize(pred_clean) == _normalize(gold_clean)


def convert_rollouts(
    rollout_dir: Path,
    output_path: Path,
    labels_path: Path,
    tokenizer_spec: str,
    expected_dim: int,
    dtype: str,
    min_attempts: int,
    uids_output: Optional[Path],
    metadata_path: Optional[Path],
    use_math_verify: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    torch = _lazy_import_torch()
    tokenizer = _lazy_tokenizer(tokenizer_spec)
    math_verify_module = _maybe_load_math_verify(use_math_verify)
    ans_map, _ = _load_math_metadata(metadata_path)

    files = list(_iter_rollout_files(rollout_dir))
    if not files:
        raise FileNotFoundError(
            f"No rollout files ending with .json.pt found in {rollout_dir}"
        )

    dtype_np = np.dtype(dtype)

    attempts = []
    for path in files:
        bundle = torch.load(path, map_location="cpu")
        hidden_tensor, hidden_dim = _as_2d_tensor(
            bundle.get("hidden_states"), torch, expected_dim
        )
        token_batch = torch.as_tensor(bundle.get("token_batch"), device="cpu")
        if token_batch.ndim != 2:
            raise ValueError(f"token_batch in {path} must be 2D")
        if token_batch.shape[0] != hidden_tensor.shape[0]:
            raise ValueError(
                f"Mismatch between token_batch rows ({token_batch.shape[0]}) and"
                f" hidden_states rows ({hidden_tensor.shape[0]}) in {path}"
            )

        prompt_ids = _coerce_prompt_ids(bundle.get("prompt"), torch)
        prompt_len = len(prompt_ids)
        problem_id = str(bundle.get("problem_id", path.stem))

        hidden_cpu = hidden_tensor.detach().to(torch.float32).cpu().numpy()

        for attempt_idx in range(token_batch.shape[0]):
            seq_tokens = token_batch[attempt_idx].tolist()
            offset = max(prompt_len - 2, 0)
            gen_tokens = seq_tokens[offset:]
            full_response = tokenizer.decode(gen_tokens, skip_special_tokens=False)
            answer_text = _extract_last_boxed(full_response).strip()
            if not answer_text:
                continue

            gold_answer = ans_map.get(problem_id)
            if gold_answer is None:
                continue

            label = int(_answer_equivalent(answer_text, gold_answer, math_verify_module))
            uid = f"{problem_id}::{path.name}::{attempt_idx:04d}"
            attempts.append(
                (problem_id, hidden_cpu[attempt_idx].astype(dtype_np, copy=False), label, uid)
            )

    if not attempts:
        raise RuntimeError("No usable attempts found across rollout files")

    counts = Counter(problem_id for problem_id, *_ in attempts)
    if min_attempts > 1:
        attempts = [rec for rec in attempts if counts[rec[0]] >= min_attempts]
        if not attempts:
            raise RuntimeError(
                "All attempts filtered out by min-attempts threshold; lower --min-attempts"
            )

    features = np.stack([rec[1] for rec in attempts], axis=0)
    labels = np.asarray([rec[2] for rec in attempts], dtype=np.int64)

    np.save(output_path, features)
    np.save(labels_path, labels)

    uids_array = None
    if uids_output is not None:
        uids_array = np.asarray([rec[3] for rec in attempts], dtype=object)
        np.save(uids_output, uids_array)

    return features, labels, uids_array


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "rollout_dir",
        type=Path,
        help="Directory containing rollout .json.pt files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("X_4096.npy"),
        help="Output path for stacked hidden vectors (default: X_4096.npy)",
    )
    parser.add_argument(
        "--labels-output",
        type=Path,
        default=Path("y_labels.npy"),
        help="Output path for binary correctness labels (default: y_labels.npy)",
    )
    parser.add_argument(
        "--uids-output",
        type=Path,
        default=None,
        help="Optional path to save attempt identifiers",
    )
    parser.add_argument(
        "--expected-dim",
        type=int,
        default=4096,
        help="Expected hidden dimension width (set 0 to disable check)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="NumPy dtype for saved feature matrix (default: float32)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Tokenizer identifier or path for decoding generations",
    )
    parser.add_argument(
        "--min-attempts",
        type=int,
        default=20,
        help="Drop problems with fewer than this many usable attempts (default: 20)",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="Optional local JSON file with MATH-500 rows (avoids HTTP fetch)",
    )
    parser.add_argument(
        "--no-math-verify",
        action="store_true",
        help="Disable math_verify and fall back to simple string comparison",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    features, labels, uids = convert_rollouts(
        rollout_dir=args.rollout_dir,
        output_path=args.output,
        labels_path=args.labels_output,
        tokenizer_spec=args.tokenizer,
        expected_dim=args.expected_dim,
        dtype=args.dtype,
        min_attempts=max(args.min_attempts, 0),
        uids_output=args.uids_output,
        metadata_path=args.metadata_json,
        use_math_verify=not args.no_math_verify,
    )

    print(
        f"Saved {features.shape[0]} vectors (width {features.shape[1]}) to {args.output}",
        file=sys.stderr,
    )
    print(
        f"Saved {labels.shape[0]} labels to {args.labels_output}",
        file=sys.stderr,
    )
    if uids is not None:
        print(
            f"Saved {uids.shape[0]} UIDs to {args.uids_output}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
