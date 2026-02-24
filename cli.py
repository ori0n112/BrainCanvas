from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from config import (
    AppConfig,
    Architecture,
    PoolMode,
    SchedulerType,
)


class Command(str, Enum):
    TRAIN = "train"
    RECONSTRUCT = "reconstruct"
    EVALUATE = "evaluate"
    SYNTHESIZE = "synthesize"


@dataclass(slots=True)
class CliRequest:
    command: Command
    config: AppConfig
    results_path: Path | None = None


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-name", type=str, default="braincanvas_stable")
    parser.add_argument("--data-root", type=Path, default=Path("../data/natural-scenes-dataset"))
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 5, 7])
    parser.add_argument("--source-subjects", type=int, nargs="+", default=[1, 2, 5])
    parser.add_argument("--target-subject", type=int, default=7)
    parser.add_argument("--adapting", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--val-batch-size", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--dataset-length", type=int, default=None)
    parser.add_argument("--pool-size", type=int, default=8192)
    parser.add_argument("--pool-mode", type=str, choices=[m.value for m in PoolMode], default=PoolMode.MAX.value)

    parser.add_argument("--architecture", type=str, choices=[a.value for a in Architecture], default=Architecture.BRAINCANVAS.value)
    parser.add_argument("--clip-variant", type=str, default="ViT-L/14")
    parser.add_argument("--normalize-clip-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--domain-hidden-size", type=int, default=1024)
    parser.add_argument("--domain-dropout", type=float, default=0.1)
    parser.add_argument("--grl-lambda", type=float, default=1.0)

    parser.add_argument("--image-weight", "--alpha", dest="image_weight", type=float, default=1.0)
    parser.add_argument("--text-weight", "--beta", dest="text_weight", type=float, default=1e4)
    parser.add_argument("--adversarial-weight", "--lambda-adv", dest="adversarial_weight", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--max-lr", type=float, default=1.5e-4)
    parser.add_argument("--scheduler", type=str, choices=[s.value for s in SchedulerType], default=SchedulerType.ONE_CYCLE.value)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=10)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=Path("../train_logs"))
    parser.add_argument("--mixed-precision", type=str, default="no")
    parser.add_argument("--deepspeed-zero-stage", type=int, default=2)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--use-wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", type=str, default="BrainCanvas")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument("--load-from", type=str, default=None)

    parser.add_argument("--append-categories-to-prompt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prompt-delimiter", type=str, default="; ")
    parser.add_argument("--max-prompt-categories", type=int, default=8)
    parser.add_argument("--use-image-augmentation", action=argparse.BooleanOptionalAction, default=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BrainCanvas v2 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(Command.TRAIN.value, help="Run model training")
    _add_shared_arguments(train_parser)

    recon_parser = subparsers.add_parser(Command.RECONSTRUCT.value, help="Run reconstruction pipeline")
    _add_shared_arguments(recon_parser)
    recon_parser.add_argument("--guidance-scale", type=float, default=5.0)
    recon_parser.add_argument("--text-image-ratio", type=float, default=0.5)
    recon_parser.add_argument("--num-inference-steps", type=int, default=20)
    recon_parser.add_argument("--recons-per-sample", type=int, default=8)
    recon_parser.add_argument("--vd-cache-dir", type=Path, default=Path("../weights"))
    recon_parser.add_argument("--test-start", type=int, default=0)
    recon_parser.add_argument("--test-end", type=int, default=None)

    eval_parser = subparsers.add_parser(Command.EVALUATE.value, help="Evaluate saved reconstructions")
    eval_parser.add_argument("--results-path", type=Path, required=True)

    synth_parser = subparsers.add_parser(Command.SYNTHESIZE.value, help="Generate class-aware synthetic dataset")
    _add_shared_arguments(synth_parser)
    synth_parser.add_argument("--work-dir", type=Path, default=None)
    synth_parser.add_argument("--sd-model-path", type=str, default="stabilityai/stable-diffusion-2-1-base")
    synth_parser.add_argument("--sd-inpainting-model-path", type=str, default=None)
    synth_parser.add_argument("--prompts-json-path", type=Path, default=Path("data/prompts/coco_prompts.json"))
    synth_parser.add_argument("--synth-batch-size", type=int, default=1)
    synth_parser.add_argument("--self-attention-resolution", type=int, default=32)
    synth_parser.add_argument("--cross-attention-resolution", type=int, default=16)
    synth_parser.add_argument("--threshold", type=float, default=0.6)
    synth_parser.add_argument("--uncertainty-threshold", type=float, default=0.5)
    synth_parser.add_argument("--start-index", type=int, default=0)
    synth_parser.add_argument("--end-index", type=int, default=None)
    synth_parser.add_argument("--diffusion-steps", type=int, default=100)
    synth_parser.add_argument(
        "--negative-prompt",
        type=str,
        default=(
            "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, "
            "out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, "
            "watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur"
        ),
    )
    synth_parser.add_argument("--attn-start-step", type=int, default=0)
    synth_parser.add_argument("--attn-end-step", type=int, default=100)
    synth_parser.add_argument("--enable-inpainting", action=argparse.BooleanOptionalAction, default=True)

    return parser


def _namespace_to_config(args: argparse.Namespace) -> AppConfig:
    cfg = AppConfig()

    cfg.runtime.run_name = args.run_name
    cfg.runtime.output_root = args.output_root
    cfg.runtime.seed = args.seed
    cfg.runtime.mixed_precision = args.mixed_precision
    cfg.runtime.deepspeed_zero_stage = args.deepspeed_zero_stage
    cfg.runtime.gradient_clip = args.gradient_clip
    cfg.runtime.use_wandb = args.use_wandb
    cfg.runtime.wandb_project = args.wandb_project
    cfg.runtime.resume = args.resume
    cfg.runtime.resume_id = args.resume_id
    cfg.runtime.load_from = args.load_from

    cfg.data.data_root = args.data_root
    cfg.data.subjects = [int(s) for s in args.subjects]
    cfg.data.source_subjects = [int(s) for s in args.source_subjects]
    cfg.data.target_subject = int(args.target_subject)
    cfg.data.adapting = args.adapting
    cfg.data.batch_size = args.batch_size
    cfg.data.val_batch_size = args.val_batch_size
    cfg.data.num_workers = args.num_workers
    cfg.data.dataset_length = args.dataset_length
    cfg.data.pool_size = args.pool_size
    cfg.data.pool_mode = PoolMode(args.pool_mode)
    cfg.data.use_image_augmentation = args.use_image_augmentation
    cfg.data.append_categories_to_prompt = args.append_categories_to_prompt
    cfg.data.prompt_delimiter = args.prompt_delimiter
    cfg.data.max_prompt_categories = args.max_prompt_categories

    cfg.model.architecture = Architecture(args.architecture)
    cfg.model.clip_variant = args.clip_variant
    cfg.model.normalize_clip_embeddings = args.normalize_clip_embeddings
    cfg.model.hidden_size = args.hidden_size
    cfg.model.depth = args.depth
    cfg.model.domain_hidden_size = args.domain_hidden_size
    cfg.model.domain_dropout = args.domain_dropout
    cfg.model.grl_lambda = args.grl_lambda

    cfg.loss.image_weight = args.image_weight
    cfg.loss.text_weight = args.text_weight
    cfg.loss.adversarial_weight = args.adversarial_weight

    cfg.optim.epochs = args.epochs
    cfg.optim.max_lr = args.max_lr
    cfg.optim.scheduler = SchedulerType(args.scheduler)
    cfg.optim.eval_every = args.eval_every
    cfg.optim.checkpoint_every = args.checkpoint_every

    if hasattr(args, "guidance_scale"):
        cfg.inference.guidance_scale = args.guidance_scale
    if hasattr(args, "text_image_ratio"):
        cfg.inference.text_image_ratio = args.text_image_ratio
    if hasattr(args, "num_inference_steps"):
        cfg.inference.num_inference_steps = args.num_inference_steps
    if hasattr(args, "recons_per_sample"):
        cfg.inference.recons_per_sample = args.recons_per_sample
    if hasattr(args, "vd_cache_dir"):
        cfg.inference.vd_cache_dir = args.vd_cache_dir
    if hasattr(args, "test_start"):
        cfg.inference.test_start = args.test_start
    if hasattr(args, "test_end"):
        cfg.inference.test_end = args.test_end

    if hasattr(args, "work_dir"):
        cfg.synthesis.work_dir = args.work_dir
    if hasattr(args, "sd_model_path"):
        cfg.synthesis.sd_model_path = args.sd_model_path
    if hasattr(args, "sd_inpainting_model_path"):
        cfg.synthesis.sd_inpainting_model_path = args.sd_inpainting_model_path
    if hasattr(args, "prompts_json_path"):
        cfg.synthesis.prompts_json_path = args.prompts_json_path
    if hasattr(args, "synth_batch_size"):
        cfg.synthesis.synth_batch_size = args.synth_batch_size
    if hasattr(args, "self_attention_resolution"):
        cfg.synthesis.self_attention_resolution = args.self_attention_resolution
    if hasattr(args, "cross_attention_resolution"):
        cfg.synthesis.cross_attention_resolution = args.cross_attention_resolution
    if hasattr(args, "threshold"):
        cfg.synthesis.threshold = args.threshold
    if hasattr(args, "uncertainty_threshold"):
        cfg.synthesis.uncertainty_threshold = args.uncertainty_threshold
    if hasattr(args, "start_index"):
        cfg.synthesis.start_index = args.start_index
    if hasattr(args, "end_index"):
        cfg.synthesis.end_index = args.end_index
    if hasattr(args, "diffusion_steps"):
        cfg.synthesis.diffusion_steps = args.diffusion_steps
    if hasattr(args, "negative_prompt"):
        cfg.synthesis.negative_prompt = args.negative_prompt
    if hasattr(args, "attn_start_step"):
        cfg.synthesis.attn_start_step = args.attn_start_step
    if hasattr(args, "attn_end_step"):
        cfg.synthesis.attn_end_step = args.attn_end_step
    if hasattr(args, "enable_inpainting"):
        cfg.synthesis.enable_inpainting = args.enable_inpainting

    return cfg


def parse_request(argv: list[str] | None = None) -> CliRequest:
    parser = build_parser()
    args = parser.parse_args(argv)

    command = Command(args.command)
    if command is Command.EVALUATE:
        return CliRequest(command=command, config=AppConfig(), results_path=args.results_path)

    config = _namespace_to_config(args)
    return CliRequest(command=command, config=config)
