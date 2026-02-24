from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Architecture(str, Enum):
    SINGLE = "single"
    FUSION = "fusion"
    BRAINCANVAS = "braincanvas"


class PoolMode(str, Enum):
    MAX = "max"
    AVG = "avg"
    RESIZE = "resize"


class SchedulerType(str, Enum):
    ONE_CYCLE = "onecycle"
    LINEAR = "linear"


@dataclass(slots=True)
class RuntimeConfig:
    run_name: str = "braincanvas_stable"
    output_root: Path = Path("../train_logs")
    seed: int = 42
    mixed_precision: str = "no"
    deepspeed_zero_stage: int = 2
    gradient_clip: float = 1.0
    use_wandb: bool = True
    wandb_project: str = "BrainCanvas"
    resume: bool = False
    resume_id: str | None = None
    load_from: str | None = None


@dataclass(slots=True)
class DataConfig:
    data_root: Path = Path("../data/natural-scenes-dataset")
    subjects: list[int] = field(default_factory=lambda: [1, 2, 5, 7])
    source_subjects: list[int] = field(default_factory=lambda: [1, 2, 5])
    target_subject: int = 7
    adapting: bool = False
    batch_size: int = 50
    val_batch_size: int = 50
    num_workers: int = 8
    pool_size: int = 8192
    pool_mode: PoolMode = PoolMode.MAX
    dataset_length: int | None = None
    use_image_augmentation: bool = True
    append_categories_to_prompt: bool = False
    prompt_delimiter: str = "; "
    max_prompt_categories: int = 8

    @property
    def active_subjects(self) -> list[int]:
        if self.adapting:
            return [*self.source_subjects, self.target_subject]
        return self.subjects


@dataclass(slots=True)
class ModelConfig:
    architecture: Architecture = Architecture.BRAINCANVAS
    clip_variant: str = "ViT-L/14"
    normalize_clip_embeddings: bool = True
    hidden_size: int = 2048
    depth: int = 4
    domain_hidden_size: int = 1024
    domain_dropout: float = 0.1
    grl_lambda: float = 1.0


@dataclass(slots=True)
class LossConfig:
    image_weight: float = 1.0
    text_weight: float = 1e4
    adversarial_weight: float = 0.1


@dataclass(slots=True)
class OptimConfig:
    epochs: int = 800
    max_lr: float = 1.5e-4
    scheduler: SchedulerType = SchedulerType.ONE_CYCLE
    eval_every: int = 10
    checkpoint_every: int = 10
    weight_decay: float = 1e-2


@dataclass(slots=True)
class InferenceConfig:
    guidance_scale: float = 5.0
    text_image_ratio: float = 0.5
    num_inference_steps: int = 20
    recons_per_sample: int = 8
    autoencoder_run_name: str | None = None
    plotting: bool = True
    verbose: bool = True
    vd_cache_dir: Path = Path("../weights")
    test_start: int = 0
    test_end: int | None = None
    sample_indices: list[int] | None = None


@dataclass(slots=True)
class SynthesisConfig:
    work_dir: Path | None = None
    sd_model_path: str = "stabilityai/stable-diffusion-2-1-base"
    sd_inpainting_model_path: str | None = None
    prompts_json_path: Path = Path("data/prompts/coco_prompts.json")
    synth_batch_size: int = 1
    self_attention_resolution: int = 32
    cross_attention_resolution: int = 16
    threshold: float = 0.6
    uncertainty_threshold: float | None = 0.5
    start_index: int = 0
    end_index: int | None = None
    diffusion_steps: int = 100
    negative_prompt: str = (
        "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, "
        "out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, "
        "watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur"
    )
    attn_start_step: int = 0
    attn_end_step: int = 100
    enable_inpainting: bool = True


@dataclass(slots=True)
class AppConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)

    @property
    def run_dir(self) -> Path:
        return self.runtime.output_root / self.runtime.run_name
