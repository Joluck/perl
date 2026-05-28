#!/usr/bin/env python
"""NanoEval entrypoint: step01 -> step02 -> step03."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanoeval.backend.runner import run_inference
from nanoeval.reward.score import eval_results
from nanoeval.utils.args import parse_cli_args, parse_task_pass_k
from nanoeval.utils.task import prepare_eval_input


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_cli_args()

    task_dir = Path(args.task_dir)
    output = Path(args.output)
    inference_output = Path(args.inference_output)
    score_output = Path(args.score_output)
    final_eval_output = Path(args.final_eval_output)

    work_dir = Path(args.work_dir) if args.work_dir else output.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    task_names, pass_k_by_task = parse_task_pass_k(
        tasks_arg=args.tasks,
        task_dir=task_dir,
        default_pass_k=args.pass_k,
    )

    # ---------- step01: prepare inputs ----------
    if args.stage in ("step01", "all"):
        logging.info("[step01] Preparing eval inputs for tasks: %s", ", ".join(task_names))
        prepare_eval_input(
            task_names=task_names,
            task_dir=task_dir,
            pass_k_by_task=pass_k_by_task,
            output_path=output,
            chat_template_model_path=args.chat_template_model_path or args.model_path,
            system_prompt=args.system_prompt,
        )
        logging.info("[step01] Saved prepared inputs to %s", output)

    # ---------- step02: inference ----------
    if args.stage in ("step02", "all"):
        input_file = Path(args.input) if args.input else output
        logging.info("[step02] Running inference with backend=%s", args.backend)
        run_inference(
            backend=args.backend,
            input_file=input_file,
            output_file=inference_output,
            sampling_params={
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "min_p": args.min_p,
                "presence_penalty": args.presence_penalty,
                "repetition_penalty": args.repetition_penalty,
                "reasoning_effort": args.reasoning_effort,
            },
            model_path=args.model_path,
            tp_size=args.tp_size,
            dp_size=args.dp_size,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            concurrency=args.concurrency,
            ray_num_actors=args.ray_num_actors,
            ray_worker_concurrency=args.ray_worker_concurrency,
            online_request_timeout_s=args.online_request_timeout_s,
            online_stall_log_interval_s=args.online_stall_log_interval_s,
        )
        logging.info("[step02] Saved inference results to %s", inference_output)

    # ---------- step03: score ----------
    if args.stage in ("step03", "all"):
        eval_input = Path(args.eval_input) if args.eval_input else inference_output
        logging.info("[step03] Scoring results from %s", eval_input)
        metrics = eval_results(
            eval_output_file=eval_input,
            score_output_file=score_output,
            final_eval_output_file=final_eval_output,
            final_eval_csv_output_file=final_eval_output.with_suffix(".csv"),
            n_proc=args.n_proc,
        )
        logging.info("[step03] Final metrics:")
        for task_name, task_metrics in metrics.items():
            logging.info("  %s — %s", task_name, task_metrics)

    logging.info("Done.")


if __name__ == "__main__":
    main()
