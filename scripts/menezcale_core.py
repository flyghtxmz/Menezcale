import re
import subprocess
import sys
from typing import Optional, Tuple

from PIL import Image

from modules import shared

try:
    import modules.sd_upscalers as sd_upscalers
except Exception:
    sd_upscalers = None

try:
    from modules import face_restoration
except Exception:
    face_restoration = None


def load_sd_parsers():
    """
    Try to import sd_parsers; if missing, attempt a lightweight pip install.
    Falls back to None if not available (regex still works).
    """
    try:
        from sd_parsers import parse_generation_parameters  # type: ignore
        return parse_generation_parameters
    except Exception:
        pass

    try:
        print("[Menezcale] Instalando dependência leve sd-parsers...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sd-parsers"])
        # Re-tentar import após instalação
        try:
            from sd_parsers import parse_generation_parameters  # type: ignore

            print("[Menezcale] sd-parsers instalado com sucesso.")
            return parse_generation_parameters
        except Exception:
            # Algumas versões expõem 'parse' em vez de parse_generation_parameters.
            try:
                import sd_parsers  # type: ignore

                if hasattr(sd_parsers, "parse"):
                    print("[Menezcale] sd-parsers instalado; usando sd_parsers.parse.")
                    return getattr(sd_parsers, "parse")
            except Exception:
                pass
    except Exception as err:
        print(f"[Menezcale] sd-parsers indisponível ({err}). Usando regex fallback.")
        return None

    print("[Menezcale] sd-parsers não expôs parse_generation_parameters; usando regex fallback.")
    return None


parse_generation_parameters = load_sd_parsers()


def log_hires_info(p):
    try:
        if not getattr(p, "enable_hr", False):
            return
        base_w, base_h = getattr(p, "width", None), getattr(p, "height", None)
        hr_scale = getattr(p, "hr_scale", None)
        hr_resize_x = getattr(p, "hr_resize_x", 0)
        hr_resize_y = getattr(p, "hr_resize_y", 0)

        target_w, target_h = None, None
        scale_used = None
        if hr_resize_x and hr_resize_y:
            target_w, target_h = hr_resize_x, hr_resize_y
        elif hr_scale and base_w and base_h:
            target_w, target_h = int(base_w * float(hr_scale)), int(base_h * float(hr_scale))
            scale_used = hr_scale

        print(
            f"[Menezcale] Hires Fix ativo. "
            f"Resolução base (antes do hires): {base_w}x{base_h}. "
            + (f"Tamanho final esperado pelo Hires: {target_w}x{target_h}. " if target_w and target_h else "")
            + (f"Fator hires: {scale_used}x." if scale_used else "")
        )
    except Exception as err:
        print(f"[Menezcale] Não foi possível registrar info do Hires Fix: {err}")


def detect_original_size(
    p,
    image: Image.Image,
    use_auto_original: bool,
    manual_width: int,
    manual_height: int,
) -> Optional[Tuple[int, int]]:
    if not use_auto_original:
        if manual_width and manual_height:
            print(f"[Menezcale] Usando tamanho manual: {manual_width}x{manual_height}")
            return int(manual_width), int(manual_height)
        return None

    info = getattr(image, "info", {}) or {}
    base_w = info.get("menezcale_base_width")
    base_h = info.get("menezcale_base_height")
    if base_w and base_h:
        try:
            bw, bh = int(base_w), int(base_h)
            print(f"[Menezcale] Tamanho original de metadata Menezcale: {bw}x{bh}")
            return bw, bh
        except Exception:
            pass

    if p and getattr(p, "width", None) and getattr(p, "height", None):
        if p.width > 0 and p.height > 0:
            print(f"[Menezcale] Tamanho original de p: {p.width}x{p.height}")
            return int(p.width), int(p.height)

    params_text = info.get("parameters") or info.get("Parameters")

    if params_text:
        match = re.search(r"Width:\s*(\d+),\s*Height:\s*(\d+)", params_text)
        if match:
            width, height = int(match.group(1)), int(match.group(2))
            print(f"[Menezcale] Tamanho original de metadados regex: {width}x{height}")
            return width, height

        size_match = re.search(r"Size:\s*(\d+)\s*[xX]\s*(\d+)", params_text)
        if size_match:
            width, height = int(size_match.group(1)), int(size_match.group(2))
            print(f"[Menezcale] Tamanho original de metadados Size: {width}x{height}")
            return width, height

        if parse_generation_parameters:
            try:
                parsed = parse_generation_parameters(params_text)
                width = parsed.get("Width") or parsed.get("width")
                height = parsed.get("Height") or parsed.get("height")
                if width and height:
                    print(f"[Menezcale] Tamanho original via sd_parsers: {width}x{height}")
                    return int(width), int(height)
            except Exception as err:
                print(f"[Menezcale] Falha ao ler metadados sd_parsers: {err}")

    print("[Menezcale] Não foi possível detectar tamanho original automaticamente.")
    return None


def compute_target_size(
    image: Image.Image,
    original_size: Optional[Tuple[int, int]],
    down_factor: float,
    use_manual_down: bool,
) -> Tuple[int, int]:
    if original_size:
        return original_size

    if use_manual_down:
        factor = max(0.01, float(down_factor or 1.0))
        width = max(1, int(image.width * factor))
        height = max(1, int(image.height * factor))
        print(f"[Menezcale] Tamanho alvo por fator manual {factor}: {width}x{height}")
        return width, height

    print("[Menezcale] Sem tamanho original detectado e fator manual desativado; mantendo tamanho atual.")
    return image.width, image.height


def apply_downscale(
    image: Image.Image,
    down_method: str,
    target_size: Tuple[int, int],
    metadata: dict,
) -> Image.Image:
    target_w, target_h = target_size
    print(
        f"[Menezcale] Downscale alvo {target_w}x{target_h} via {down_method}"
    )

    method_key = down_method.lower()

    if "lanczos" in method_key:
        result = image.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)
    elif "bicubic" in method_key:
        result = image.resize((target_w, target_h), resample=Image.Resampling.BICUBIC)
    elif "fsrcnn" in method_key:
        result = downscale_with_fsrcnn(image, target_w, target_h, metadata)
    else:
        result = image.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)

    result.info = metadata.copy()
    return result


def downscale_with_fsrcnn(
    image: Image.Image,
    target_w: int,
    target_h: int,
    metadata: dict,
) -> Image.Image:
    upscaler = find_fsrcnn_upscaler()
    scale = min(target_w / image.width, target_h / image.height)
    scale = max(scale, 0.01)

    if upscaler:
        try:
            print(
                f"[Menezcale] Usando FSRCNN com fator aproximado {scale:.3f}"
            )
            result = upscaler.upscale(image, scale)
            if result:
                result = result.resize((target_w, target_h), Image.Resampling.LANCZOS)
                result.info = metadata.copy()
                return result
        except Exception as err:
            print(f"[Menezcale] Falha FSRCNN ({err}). Fallback para Lanczos.")

    result = image.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)
    result.info = metadata.copy()
    return result


def apply_face_restore_if_enabled(
    image: Image.Image,
    metadata: dict,
) -> Image.Image:
    if not face_restoration:
        return image

    model_name = getattr(shared.opts, "face_restoration_model", None)
    if not model_name:
        return image

    try:
        restored = face_restoration.restore_faces(image)
        if isinstance(restored, tuple):
            restored = restored[0]
        if restored is not None:
            restored.info = metadata.copy()
            print(f"[Menezcale] GFPGAN/face restoration aplicado ({model_name}).")
            return restored
    except Exception as err:
        print(f"[Menezcale] Face restoration falhou ({err}).")
    return image


def find_upscaler_by_name(name: str):
    if not sd_upscalers:
        return None
    try:
        if hasattr(sd_upscalers, "get_upscalers"):
            for upscaler in sd_upscalers.get_upscalers():
                if getattr(upscaler, "name", "").lower() == name.lower():
                    return upscaler
        if hasattr(sd_upscalers, "upscaler_for_name"):
            upscaler = sd_upscalers.upscaler_for_name(name)
            if upscaler:
                return upscaler
    except Exception:
        return None
    return None


def find_fsrcnn_upscaler():
    if not sd_upscalers:
        return None
    try:
        if hasattr(sd_upscalers, "get_upscalers"):
            for upscaler in sd_upscalers.get_upscalers():
                if "fsrcnn" in getattr(upscaler, "name", "").lower():
                    return upscaler
    except Exception:
        pass

    # Try loading by common model name.
    return find_upscaler_by_name("FSRCNN_x2")


def safe_copy_image(image: Image.Image) -> Image.Image:
    try:
        img = image.copy()
        img.info = getattr(image, "info", {}).copy()
        return img
    except Exception:
        return image


def attach_base_metadata(
    metadata: dict,
    original_size: Optional[Tuple[int, int]],
    p,
):
    try:
        if original_size:
            metadata["menezcale_base_width"] = int(original_size[0])
            metadata["menezcale_base_height"] = int(original_size[1])
            metadata["menezcale_hires_enabled"] = True
            return
        if p and getattr(p, "width", None) and getattr(p, "height", None):
            metadata["menezcale_base_width"] = int(p.width)
            metadata["menezcale_base_height"] = int(p.height)
            metadata["menezcale_hires_enabled"] = bool(getattr(p, "enable_hr", False))
    except Exception:
        pass


def is_hires_allowed(image: Optional[Image.Image], hires_available_flag: bool = False) -> bool:
    if hires_available_flag:
        return True
    if image is None:
        return False
    info = getattr(image, "info", {}) or {}
    if info.get("menezcale_hires_enabled"):
        return True
    params_text = info.get("parameters") or info.get("Parameters")
    if params_text and re.search(r"hires", params_text, re.IGNORECASE):
        return True
    return False
