from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cli import Command, parse_request
from config import AppConfig
from data import COCOTextRepository, build_loader_bundle
from data.dataset import NSDRecordDataset
from diffusion import ImageSetSynthesizer, load_prompt_records
from engine import Trainer, create_batch_stream
from engine.image_metrics import evaluate_image_metrics
from models import ClipFeatureExtractor, build_decoder
from runtime import create_accelerator
from utils import seed_everything


def _build_training_components(config: AppConfig):
    accelerator = create_accelerator(config.runtime)
    seed_everything(config.runtime.seed, deterministic_cudnn=False)

    adjusted_lr = config.optim.max_lr * accelerator.num_processes
    config.optim.max_lr = adjusted_lr

    clip_model = ClipFeatureExtractor(
        variant=config.model.clip_variant,
        device=accelerator.device,
        normalize_embeddings=config.model.normalize_clip_embeddings,
        use_hidden_state=True,
    )

    model = build_decoder(
        config.model,
        subjects=config.data.active_subjects,
        in_dim=config.data.pool_size,
        image_out_dim=clip_model.image_output_dim,
        text_out_dim=clip_model.text_output_dim,
        adapting=config.data.adapting,
    ).to(accelerator.device)

    loaders = build_loader_bundle(config.data, config.model.architecture, config.runtime.seed)
    stream = create_batch_stream(loaders)

    prompt_repo = COCOTextRepository(
        config.data.data_root,
        append_categories=config.data.append_categories_to_prompt,
        delimiter=config.data.prompt_delimiter,
        max_categories=config.data.max_prompt_categories,
    )

    return accelerator, model, clip_model, stream, prompt_repo


def run_training(config: AppConfig) -> None:
    accelerator, model, clip_model, stream, prompt_repo = _build_training_components(config)
    trainer = Trainer(
        config=config,
        accelerator=accelerator,
        model=model,
        clip_extractor=clip_model,
        batch_stream=stream,
        prompt_repo=prompt_repo,
    )
    trainer.fit()


def run_reconstruction(config: AppConfig) -> None:
    seed_everything(config.runtime.seed, deterministic_cudnn=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subject = config.data.target_subject if config.data.adapting else config.data.subjects[0]
    test_root = config.data.data_root / "webdataset_avg_split" / "test" / f"subj0{subject}"

    dataset = NSDRecordDataset(
        test_root,
        pool_size=config.data.pool_size,
        pool_mode=config.data.pool_mode,
        length=None,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=config.data.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    clip_model = ClipFeatureExtractor(
        variant=config.model.clip_variant,
        device=device,
        normalize_embeddings=config.model.normalize_clip_embeddings,
        use_hidden_state=True,
    )
    model = build_decoder(
        config.model,
        subjects=config.data.active_subjects,
        in_dim=config.data.pool_size,
        image_out_dim=clip_model.image_output_dim,
        text_out_dim=clip_model.text_output_dim,
        adapting=config.data.adapting,
    ).to(device)

    checkpoint_path = config.runtime.load_from or str(config.run_dir / "last.pt")
    payload = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()

    def _anchor_from_tokens(tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() == 3:
            return tokens[:, 0, :]
        return tokens.flatten(1)

    prompt_repo: COCOTextRepository | None = None
    try:
        prompt_repo = COCOTextRepository(
            config.data.data_root,
            append_categories=config.data.append_categories_to_prompt,
            delimiter=config.data.prompt_delimiter,
            max_categories=config.data.max_prompt_categories,
        )
    except Exception as exc:
        print(f"Text-guided reconstruction fallback: prompt repository unavailable ({exc})")

    out_dir = config.run_dir / f"reconstruction_subj{subject}"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths: list[Path] = []
    target_image_anchor_bank: list[torch.Tensor] = []
    pred_image_anchor_bank: list[torch.Tensor] = []
    target_text_anchor_bank: list[torch.Tensor] = []
    pred_text_anchor_bank: list[torch.Tensor] = []

    with torch.no_grad():
        for index, item in enumerate(loader):
            voxel = item["voxel"].to(device).float()
            subject_id = item["subject_id"].to(device).long().view(-1)
            image = item["image"].float().cpu()

            if voxel.dim() >= 3:
                voxel = voxel.mean(dim=1)

            output = model(voxel, subject_id)
            image_path = out_dir / f"{index}_img.pt"
            torch.save(image, image_path)
            torch.save(output.image_embedding.cpu(), out_dir / f"{index}_img_embed.pt")
            torch.save(output.text_embedding.cpu(), out_dir / f"{index}_txt_embed.pt")

            target_image_tokens = clip_model.encode_images(image.to(device))
            pred_image_tokens = output.image_embedding.reshape(output.image_embedding.shape[0], -1, clip_model.width)
            target_image_anchor = _anchor_from_tokens(target_image_tokens)
            pred_image_anchor = _anchor_from_tokens(pred_image_tokens)

            target_image_anchor_bank.append(torch.nn.functional.normalize(target_image_anchor.float(), dim=-1).cpu())
            pred_image_anchor_bank.append(torch.nn.functional.normalize(pred_image_anchor.float(), dim=-1).cpu())

            if prompt_repo is not None:
                prompts = prompt_repo.prompts_for_batch(item["coco_id"], repeat_index=0)
                target_text_tokens = clip_model.encode_texts(prompts)
                pred_text_tokens = output.text_embedding.reshape(output.text_embedding.shape[0], -1, clip_model.width)
                target_text_anchor = _anchor_from_tokens(target_text_tokens)
                pred_text_anchor = _anchor_from_tokens(pred_text_tokens)

                target_text_anchor_bank.append(torch.nn.functional.normalize(target_text_anchor.float(), dim=-1).cpu())
                pred_text_anchor_bank.append(torch.nn.functional.normalize(pred_text_anchor.float(), dim=-1).cpu())
            image_paths.append(image_path)

    if len(pred_image_anchor_bank) == 0:
        print(f"Reconstruction export complete: {out_dir}")
        return

    target_image_matrix = torch.cat(target_image_anchor_bank, dim=0)
    pred_image_matrix = torch.cat(pred_image_anchor_bank, dim=0)

    text_guidance_ready = len(target_text_anchor_bank) == len(target_image_anchor_bank) and len(target_text_anchor_bank) > 0
    text_ratio = float(config.inference.text_image_ratio)
    text_ratio = min(1.0, max(0.0, text_ratio))
    guidance_scale = max(0.0, float(config.inference.guidance_scale))

    image_similarity = pred_image_matrix @ target_image_matrix.T
    if text_guidance_ready:
        target_text_matrix = torch.cat(target_text_anchor_bank, dim=0)
        pred_text_matrix = torch.cat(pred_text_anchor_bank, dim=0)
        text_similarity = pred_text_matrix @ target_text_matrix.T
        conditioned_similarity = (1.0 - text_ratio) * image_similarity + text_ratio * text_similarity
    else:
        conditioned_similarity = image_similarity

    guided_similarity = image_similarity + guidance_scale * (conditioned_similarity - image_similarity)

    nearest_indices: list[int] = []
    chunk_size = 512
    for start in range(0, guided_similarity.shape[0], chunk_size):
        guided_chunk = guided_similarity[start : start + chunk_size]
        nearest_indices.extend(guided_chunk.argmax(dim=1).tolist())

    cache: dict[int, torch.Tensor] = {}
    for index, nearest in enumerate(nearest_indices):
        if nearest not in cache:
            cache[nearest] = torch.load(image_paths[nearest], map_location="cpu")
        torch.save(cache[nearest], out_dir / f"{index}_recon.pt")

    print(f"Reconstruction export complete: {out_dir}")
    print(f"Generated {len(nearest_indices)} retrieval-based reconstruction files (`*_recon.pt`) for 8-metric evaluation")
    print(
        "Reconstruction guidance settings: "
        f"guidance_scale={guidance_scale:.3f}, "
        f"text_image_ratio={text_ratio:.3f}, "
        f"text_guidance={'on' if text_guidance_ready else 'off'}"
    )


def run_evaluation(config: AppConfig, results_path: Path) -> None:
    if not results_path.exists():
        raise FileNotFoundError(f"results_path does not exist: {results_path}")

    image_files = sorted(results_path.glob("*_img.pt"))
    embed_files = sorted(results_path.glob("*_img_embed.pt"))
    if len(image_files) > 0 and len(embed_files) > 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clip_model = ClipFeatureExtractor(
            variant=config.model.clip_variant,
            device=device,
            normalize_embeddings=config.model.normalize_clip_embeddings,
            use_hidden_state=True,
        )

        pair_count = min(len(image_files), len(embed_files))
        cosine_scores: list[float] = []

        with torch.no_grad():
            for index in range(pair_count):
                image = torch.load(image_files[index], map_location=device).float()
                pred = torch.load(embed_files[index], map_location=device).float()
                target = clip_model.encode_images(image)
                pred = torch.nn.functional.normalize(pred.flatten(1), dim=-1)
                target = torch.nn.functional.normalize(target.flatten(1), dim=-1)
                cosine = torch.nn.functional.cosine_similarity(pred, target).mean().item()
                cosine_scores.append(cosine)

        print(f"Evaluated {pair_count} embedding pairs")
        print(f"Image embedding cosine mean: {float(np.mean(cosine_scores)):.6f}")
    else:
        print(
            "Embedding cosine evaluation skipped: "
            f"found {len(image_files)} files matching `*_img.pt` and {len(embed_files)} files matching `*_img_embed.pt`"
        )

    try:
        metrics, pair_count = evaluate_image_metrics(results_path)
    except RuntimeError as exc:
        print(f"8-metric image evaluation skipped: {exc}")
        return

    print(f"Evaluated {pair_count} image reconstruction pairs")
    print("8-metric protocol results:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


def run_synthesis(config: AppConfig) -> None:
    seed_everything(config.runtime.seed, deterministic_cudnn=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = load_prompt_records(config.synthesis.prompts_json_path)
    output_dir = config.synthesis.work_dir if config.synthesis.work_dir is not None else (config.run_dir / "synthetic")

    synthesizer = ImageSetSynthesizer(
        sd_model_path=config.synthesis.sd_model_path,
        sd_inpainting_model_path=config.synthesis.sd_inpainting_model_path,
        device=device,
        seed=config.runtime.seed,
        self_attention_resolution=config.synthesis.self_attention_resolution,
        cross_attention_resolution=config.synthesis.cross_attention_resolution,
        threshold=config.synthesis.threshold,
        uncertainty_threshold=config.synthesis.uncertainty_threshold,
        start_step=config.synthesis.attn_start_step,
        end_step=config.synthesis.attn_end_step,
        inpaint_enabled=config.synthesis.enable_inpainting,
    )
    synthesizer.synthesize(
        records=records,
        output_dir=output_dir,
        batch_size=config.synthesis.synth_batch_size,
        num_inference_steps=config.synthesis.diffusion_steps,
        negative_prompt=config.synthesis.negative_prompt,
        start_index=config.synthesis.start_index,
        end_index=config.synthesis.end_index,
    )
    print(f"Synthesis complete. Output directory: {output_dir}")


def main(argv: list[str] | None = None) -> None:
    request = parse_request(argv)

    if request.command is Command.TRAIN:
        run_training(request.config)
        return

    if request.command is Command.RECONSTRUCT:
        run_reconstruction(request.config)
        return

    if request.command is Command.EVALUATE:
        assert request.results_path is not None
        run_evaluation(request.config, request.results_path)
        return

    if request.command is Command.SYNTHESIZE:
        run_synthesis(request.config)
        return

    raise RuntimeError(f"Unhandled command: {request.command}")


if __name__ == "__main__":
    main()
