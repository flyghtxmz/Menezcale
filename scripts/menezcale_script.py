import re
import subprocess
import sys
from typing import Optional, Tuple

import gradio as gr
from PIL import Image

import modules.scripts as scripts
import modules.shared as shared
from modules.processing import Processed, StableDiffusionProcessing

try:
    import modules.sd_upscalers as sd_upscalers
except Exception:
    sd_upscalers = None


def _load_sd_parsers():
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


# Optional dependency requested by spec for parsing metadata.
parse_generation_parameters = _load_sd_parsers()

try:
    from modules import face_restoration
except Exception:
    face_restoration = None


class MenezcaleScript(scripts.Script):
    """
    Menezcale - Auto Downscale (foco em nitidez após upscale externo/Hires).

    O script detecta o tamanho original (incluindo Hires Fix) e permite
    downscale para esse tamanho. Pode rodar automático no postprocess ou
    manualmente pelo painel (com preview).
    """
    _hires_available: bool = False
    _last_image: Optional[Image.Image] = None

    def title(self):
        return "Menezcale"

    def show(self, is_img2img):
        # Visible only on txt2img per spec.
        return scripts.AlwaysVisible if not is_img2img else None

    def ui(self, is_img2img):
        with gr.Accordion(
            "Menezcale - Auto Downscale (Foco em Nitidez Pós-Upscale)",
            open=False,
        ):
            down_method = gr.Dropdown(
                choices=[
                    "Lanczos (Recomendado para Preservar Qualidade)",
                    "FSRCNN (IA para Downscale Inteligente)",
                    "Bicubic (Rápido)",
                ],
                value="Lanczos (Recomendado para Preservar Qualidade)",
                label="Método de Downscale",
            )

            use_auto_original = gr.Checkbox(
                label="Tamanho original",
                value=True,
                info="Quando ligado, volta para o tamanho original detectado (p.width/p.height, Hires Fix ou metadados).",
            )

            use_manual_down = gr.Checkbox(
                label="Usar fator manual de downscale (opcional)",
                value=False,
                info="Marque para habilitar o fator manual; caso contrário usa o tamanho original.",
            )

            down_factor = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                step=0.05,
                value=0.25,
                label="Fator de Downscale (manual)",
                visible=False,
            )

            manual_width = gr.Slider(
                minimum=64,
                maximum=4096,
                step=1,
                value=512,
                label="Width Manual",
                visible=False,
            )

            manual_height = gr.Slider(
                minimum=64,
                maximum=4096,
                step=1,
                value=512,
                label="Height Manual",
                visible=False,
            )

            gr.Markdown("### Teste Manual de Downscale (pós-upscale)")

            manual_input = gr.Image(
                label="Imagem pós-upscale (upload)",
                type="pil",
                height=256,
            )
            load_last = gr.Button("Carregar última imagem gerada")
            manual_button = gr.Button("Aplicar Downscale")
            manual_output = gr.Image(
                label="Preview Downscale",
                type="pil",
                height=256,
            )

            # UI interatividade.
            use_auto_original.change(
                fn=lambda enabled: (
                    gr.update(visible=not enabled),
                    gr.update(visible=not enabled),
                ),
                inputs=use_auto_original,
                outputs=[manual_width, manual_height],
            )

            use_manual_down.change(
                fn=lambda enabled: gr.update(visible=enabled),
                inputs=use_manual_down,
                outputs=down_factor,
            )

            manual_button.click(
                fn=self._manual_test,
                inputs=[
                    manual_input,
                    down_method,
                    down_factor,
                    use_manual_down,
                    use_auto_original,
                    manual_width,
                    manual_height,
                ],
                outputs=manual_output,
            )

            load_last.click(
                fn=self._load_last_image,
                inputs=[],
                outputs=[manual_input, manual_output],
            )

        return [
            down_method,
            down_factor,
            use_manual_down,
            use_auto_original,
            manual_width,
            manual_height,
        ]

    def _manual_test(
        self,
        image: Optional[Image.Image],
        down_method: str,
        down_factor: float,
        use_manual_down: bool,
        use_auto_original: bool,
        manual_width: int,
        manual_height: int,
    ) -> Optional[Image.Image]:
        if image is None:
            return None
        if not self._is_hires_allowed(image):
            print("[Menezcale] Hires Fix não detectado. Downscale bloqueado.")
            return None

        print("[Menezcale] Teste manual iniciado")
        processed_image = self._run_pipeline(
            image=image,
            p=None,
            down_method=down_method,
            down_factor=down_factor,
            use_manual_down=use_manual_down,
            use_auto_original=use_auto_original,
            manual_width=manual_width,
            manual_height=manual_height,
        )
        print("[Menezcale] Teste manual concluído")
        return processed_image

    def postprocess(
        self,
        p: StableDiffusionProcessing,
        processed: Processed,
        down_method: str,
        down_factor: float,
        use_manual_down: bool,
        use_auto_original: bool,
        manual_width: int,
        manual_height: int,
    ):
        self._log_hires_info_if_any(p)
        hires_enabled = bool(getattr(p, "enable_hr", False))
        self._hires_available = hires_enabled

        if not processed or not processed.images:
            print("[Menezcale] Nenhuma imagem processada encontrada.")
            return

        # Sempre guarda a última imagem gerada para o botão de carregamento.
        self._last_image = self._safe_copy_image(processed.images[-1])
        self._attach_base_metadata(
            getattr(self._last_image, "info", {}),
            original_size=None,
            p=p,
        )

        if not hires_enabled:
            print("[Menezcale] Hires Fix não está ativo; controles bloqueados até habilitar Hires.")
            return
        # Automático desativado: apenas registra a última imagem e sai.
        print("[Menezcale] Downscale automático desativado. Use o botão 'Aplicar Downscale'.")

    def _load_last_image(self):
        img = getattr(self, "_last_image", None)
        if img is None:
            print("[Menezcale] Nenhuma imagem gerada anteriormente para carregar.")
            return None, None
        if not self._is_hires_allowed(img):
            print("[Menezcale] Hires Fix não detectado na última imagem; controles bloqueados.")
            return None, None
        try:
            copy_img = img.copy()
            copy_img.info = getattr(img, "info", {}).copy()
            return copy_img, copy_img
        except Exception as err:
            print(f"[Menezcale] Falha ao carregar última imagem: {err}")
            return None, None

    def _log_hires_info_if_any(self, p: StableDiffusionProcessing):
        """
        If Hires Fix is enabled in txt2img, log the base resolution before hires.
        This helps the user know the original size prior to the upscale.
        """
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

    # Core processing helpers -------------------------------------------------
    def _run_pipeline(
        self,
        image: Image.Image,
        p: Optional[StableDiffusionProcessing],
        down_method: str,
        down_factor: float,
        use_manual_down: bool,
        use_auto_original: bool,
        manual_width: int,
        manual_height: int,
    ) -> Image.Image:
        original_info = getattr(image, "info", {}) or {}

        original_size = self._detect_original_size(
            p=p,
            image=image,
            use_auto_original=use_auto_original,
            manual_width=manual_width,
            manual_height=manual_height,
        )

        self._attach_base_metadata(original_info, original_size, p)

        target_size = self._compute_target_size(
            image=image,
            original_size=original_size,
            down_factor=down_factor,
            use_manual_down=use_manual_down,
        )

        image = self._apply_downscale(image, down_method, target_size, original_info)
        image = self._apply_face_restore_if_enabled(image, original_info)

        print(
            f"Menezcale: Downscale para {target_size[0]}x{target_size[1]} aplicado "
            f"com {down_method} "
            f"(alvo {'original' if original_size else ('fator manual' if use_manual_down else 'tamanho atual')})"
        )

        # Reattach metadata for downstream consumers.
        image.info.update(original_info)
        return image

    def _detect_original_size(
        self,
        p: Optional[StableDiffusionProcessing],
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

    def _compute_target_size(
        self,
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

    def _apply_upscale(
        self,
        image: Image.Image,
        upscale_method: str,
        upscale_factor: float,
        metadata: dict,
    ) -> Image.Image:
        scale = max(1.0, float(upscale_factor or 1.0))
        target_size = (int(image.width * scale), int(image.height * scale))
        print(
            f"[Menezcale] Upscale automático: método {upscale_method} "
            f"x{scale} -> alvo {target_size}"
        )

        upscaler = self._find_upscaler_by_name(upscale_method)
        if upscaler:
            try:
                result = upscaler.upscale(image, scale)
                if result:
                    result.info = metadata.copy()
                    return result
            except Exception as err:
                print(f"[Menezcale] Upscaler '{upscale_method}' falhou ({err}). Usando fallback.")

        resized = image.resize(target_size, resample=Image.Resampling.LANCZOS)
        resized.info = metadata.copy()
        return resized

    def _apply_downscale(
        self,
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
            result = self._downscale_with_fsrcnn(image, target_w, target_h, metadata)
        else:
            result = image.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)

        result.info = metadata.copy()
        return result

    def _downscale_with_fsrcnn(
        self,
        image: Image.Image,
        target_w: int,
        target_h: int,
        metadata: dict,
    ) -> Image.Image:
        upscaler = self._find_fsrcnn_upscaler()
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

    def _apply_face_restore_if_enabled(
        self,
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

    def _find_upscaler_by_name(self, name: str):
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

    def _find_fsrcnn_upscaler(self):
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
        return self._find_upscaler_by_name("FSRCNN_x2")

    def _safe_copy_image(self, image: Image.Image) -> Image.Image:
        try:
            img = image.copy()
            img.info = getattr(image, "info", {}).copy()
            return img
        except Exception:
            return image

    def _attach_base_metadata(
        self,
        metadata: dict,
        original_size: Optional[Tuple[int, int]],
        p: Optional[StableDiffusionProcessing],
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

    def _is_hires_allowed(self, image: Optional[Image.Image]) -> bool:
        if getattr(self, "_hires_available", False):
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
