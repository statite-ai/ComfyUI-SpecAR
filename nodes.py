# SPDX-License-Identifier: Apache-2.0
"""SpecAR text-to-video node. It calls the installed `specar` package and patches it only at runtime, in this process."""
import gc
import importlib.util
import logging
import os
import stat
import sys
import time

import comfy.model_management as mm
import comfy.utils
import folder_paths
import torch

FPS = 16.0
HIGH_VRAM_GIB = 75    # an "80 GB" H100 reports about 79.6 GiB
_loaded = None        # (weights, low_vram, model): one model set stays loaded between runs
_load_count = 0


def _pinned(tensor):
    cpu = tensor.to("cpu")
    try:
        return cpu.pin_memory()
    except RuntimeError:  # the machine does not allow this much pinned memory; plain RAM only copies slower
        return cpu


class _Swap:
    """Keeps a module's weights in pinned RAM and copies them to the GPU only while the module is active.

    Weights never change during generation, so eviction drops the GPU copy instead of copying it back. Modules in the
    same group are exclusive: activating one evicts the others.
    """

    def __init__(self, module, group=()):
        self.module, self.group, self.active = module, group, False
        self.params = {}
        self.buffers = {}
        with torch.inference_mode(False):
            for m in module.modules():
                for p in m._parameters.values():
                    if p is not None and id(p) not in self.params:
                        self.params[id(p)] = (p, _pinned(p.data))
                        p.data = self.params[id(p)][1]
                for b in m._buffers.values():
                    if b is not None:
                        self.buffers[id(b)] = b
                        b.data = b.data.to("cpu")
        torch.cuda.empty_cache()

    def activate(self):
        for other in self.group:
            if other is not self:
                other.evict()
        if self.active:
            return
        with torch.inference_mode(False):  # hooks fire inside the session's inference_mode
            for p, cpu in self.params.values():
                p.data = cpu.to("cuda", non_blocking=True)
            for b in self.buffers.values():
                b.data = b.data.to("cuda")
        self.active = True

    def evict(self):
        if not self.active:
            return
        with torch.inference_mode(False):
            for p, cpu in self.params.values():
                p.data = cpu
            for b in self.buffers.values():
                b.data = b.data.to("cpu")
        self.active = False
        torch.cuda.empty_cache()


def _exclusive(modules):
    """Offload `modules` so that calling one moves it to the GPU and moves the others out."""
    group = []
    for module in modules:
        swap = _Swap(module, group)
        group.append(swap)
        module.register_forward_pre_hook(lambda _m, _args, s=swap: s.activate())
    return group


def _free():
    global _loaded
    _loaded = None
    gc.collect()
    torch.cuda.empty_cache()


def _import_krea(weights):
    """Import the Krea code without changing what `import utils` means for ComfyUI or any other custom node.

    ComfyUI's top-level `utils` is imported first, and the Krea code asks for `utils.scheduler` and `utils.wan_wrapper`.
    Registering exactly those two names satisfies Krea and leaves ComfyUI's package and its search path untouched.
    """
    import specar
    vendor = os.path.join(os.path.dirname(specar.__file__), "_vendor", "krea")
    if vendor not in sys.path:
        sys.path.append(vendor)  # last, so these modules cannot shadow anything; model_krea skips its own insert
    os.environ["MODEL_FOLDER"] = weights
    import specar.model_krea as krea
    for name in ("scheduler", "wan_wrapper"):
        full = f"utils.{name}"
        if full not in sys.modules:
            spec = importlib.util.spec_from_file_location(full, os.path.join(vendor, "utils", f"{name}.py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[full] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                del sys.modules[full]
                raise
    # wan_wrapper copies MODEL_FOLDER at import, so a new weights folder has to be written back to both modules.
    sys.modules["settings"].MODEL_FOLDER = weights
    sys.modules["utils.wan_wrapper"].MODEL_FOLDER = weights
    return krea


def _load_krea(weights, low_vram):
    krea = _import_krea(weights)
    load_text_encoder, load_wan_wrapper = krea.load_text_encoder, krea.load_wan_wrapper
    swaps = {}

    def load_t5_off_gpu():  # encoding takes a moment, so T5 should not hold GPU memory for the whole video
        text_encoder = load_text_encoder()
        swaps["t5"] = _Swap(text_encoder)
        return text_encoder

    def load_transformer(*args, **kw):
        krea.DEV = "cpu" if low_vram else "cuda"
        try:
            wrapper = load_wan_wrapper(*args, **kw)
        finally:
            krea.DEV = "cuda"
        for block in wrapper.model.blocks:  # fuse_projections copied these into to_qkv and forward only reads to_qkv
            del block.self_attn.q, block.self_attn.k, block.self_attn.v
        torch.cuda.empty_cache()
        return wrapper

    krea.load_text_encoder, krea.load_wan_wrapper = load_t5_off_gpu, load_transformer
    try:
        model = krea.Models(weights)
    finally:
        krea.load_text_encoder, krea.load_wan_wrapper = load_text_encoder, load_wan_wrapper
    model.t5_swap = swaps["t5"]
    if low_vram:
        _exclusive([model.target, model.drafter])
    return model


def _model(weights, low_vram):
    global _loaded, _load_count
    key = (weights, low_vram)
    if _loaded and _loaded[:2] == key:
        return _loaded[2]
    _free()
    _load_count += 1
    logging.info(f"SpecAR load #{_load_count}: {weights}, low_vram_offload={low_vram}")
    model = _load_krea(weights, low_vram)
    _loaded = (*key, model)
    return model


def _generate(model, prompt, anchors, blocks, seed, tick):
    from specar.model_krea import SpecARSession
    if getattr(model, "embeds", (None,))[0] != prompt:
        start = time.perf_counter()
        model.t5_swap.activate()
        embeds = model.encode_prompts([prompt])[prompt]
        model.t5_swap.evict()
        model.embeds = (prompt, embeds)
        logging.info(f"SpecAR prompt encoded in {time.perf_counter() - start:.3f} s")
    frames = []

    def on_frames(pixels, event):  # same conversion as specar.cli.run and write_mp4
        event.synchronize()
        frames.append(pixels.cpu().add_(1.0).mul_(0.5).clamp_(0.0, 1.0))

    session = SpecARSession(model, prompt, seed, blocks, anchors, prompt_embeds=model.embeds[1], frame_callback=on_frames)
    for _ in range(blocks):
        session.generate_block()
        tick()
    return (torch.cat(frames, dim=1)[0].permute(0, 2, 3, 1) * 255).to(torch.uint8), session.all_latents


WEIGHTS_ROOT = os.path.join(folder_paths.models_dir, "specar")
folder_paths.add_model_folder_path("specar", WEIGHTS_ROOT, is_default=True)


def _plain_dir(entry):
    # Symlinks and Windows junctions are skipped: they could lead outside models/specar, even to a network share.
    if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
        return False
    return not getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT


def _weights_dirs():
    """Weights folders by name, from every models/specar path (including extra_model_paths.yaml); the first path wins."""
    dirs = {}
    for root in folder_paths.get_folder_paths("specar"):
        if os.path.isdir(root):
            with os.scandir(root) as entries:
                for entry in entries:
                    if _plain_dir(entry):
                        dirs.setdefault(entry.name, entry.path)
    return dirs


def _weights_choices():
    # krea is always listed so a missing download gets a clear error.
    return ["krea"] + sorted(set(_weights_dirs()) - {"krea"})


def _weights_path(name):
    # Only a folder found by _weights_dirs reaches the loader, so a typed path or network share never does.
    path = _weights_dirs().get(name)
    if path is None:
        if name != "krea":
            raise ValueError(f"Unknown SpecAR weights {name!r}; pick a folder under models/specar")
        path = os.path.join(WEIGHTS_ROOT, "krea")
        raise FileNotFoundError(f"No weights in {path}. Run: specar download --backbone krea --weights {path}")
    return path


class SpecARGenerate:
    CATEGORY = "SpecAR"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("images", "fps")

    @classmethod
    def INPUT_TYPES(cls):
        small_gpu = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < HIGH_VRAM_GIB * 2**30
        return {"required": {
            "prompt": ("STRING", {"multiline": True, "default": "A papercraft coral reef full of colorful fish"}),
            "k": ("INT", {"default": 7, "min": 1, "max": 999, "display_name": "k (quality)", "tooltip": "How much of the video the large model generates. Higher is slower; 2 is the fastest setting."}),
            "seconds": ("FLOAT", {"default": 26.6, "min": 0.5, "max": 600.0, "step": 0.1, "tooltip": "Video length, rounded to the nearest 0.75 s. 26.6 is the tested length."}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            "weights": (_weights_choices(), {"default": "krea", "tooltip": "A folder under models/specar."}),
            "keep_loaded": ("BOOLEAN", {"default": True, "tooltip": "Keep the models loaded for the next run."}),
            "low_vram_offload": ("BOOLEAN", {"default": small_gpu, "tooltip": "Keep only the active model on the GPU."}),
        }}

    def generate(self, prompt, k, seconds, seed, weights, keep_loaded, low_vram_offload):
        from specar import uniform_k
        from specar.cli import blocks_for_seconds
        if not prompt.strip():
            raise ValueError("SpecAR needs a prompt")
        blocks = blocks_for_seconds("krea", seconds)
        k = min(k, blocks)  # a short video has fewer blocks than k
        anchors = uniform_k(k, blocks)
        weights = _weights_path(weights)

        mm.unload_all_models()
        torch.cuda.empty_cache()
        bar = comfy.utils.ProgressBar(blocks)

        def tick():
            bar.update(1)
            mm.throw_exception_if_processing_interrupted()

        # ComfyUI runs nodes under inference_mode. The specar CLI loads and sets up with gradients off instead, and
        # tensors made under inference_mode cannot be changed in place later, so match the CLI here.
        with torch.inference_mode(False), torch.no_grad():
            model = _model(weights, low_vram_offload)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            frames, latents = _generate(model, prompt, anchors, blocks, seed, tick)
            elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 2**30
        logging.info(f"SpecAR: {len(frames)} frames, k={k}, seed={seed}, generated in {elapsed:.1f} s, "
              f"peak GPU memory {peak:.1f} GiB, loads so far {_load_count}")
        del model, latents
        if keep_loaded:
            torch.cuda.empty_cache()
        else:
            _free()
        return (frames.float() / 255.0, FPS)


NODE_CLASS_MAPPINGS = {"SpecARGenerate": SpecARGenerate}
NODE_DISPLAY_NAME_MAPPINGS = {"SpecARGenerate": "SpecAR Generate"}
