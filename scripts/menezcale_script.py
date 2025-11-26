import os
import sys
from typing import Optional

import gradio as gr
from PIL import Image

import modules.scripts as scripts
from modules.processing import Processed, StableDiffusionProcessing

# Garantir que os módulos auxiliares locais sejam importáveis.
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from menezcale_core import (
    apply_downscale,
    apply_face_restore_if_enabled,
    attach_base_metadata,
    compute_target_size,
    detect_original_size,
    is_hires_allowed,
    log_hires_info,
    safe_copy_image,
)


class MenezcaleScript(scripts.Script):
    """
    Menezcale - Auto Downscale (foco em nitidez após upscale externo/Hires).

    O script detecta o tamanho original (incluindo Hires Fix) e permite
    downscale para esse tamanho. Fluxo manual pelo painel (com preview),
    usando a última imagem gerada.
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
        if not is_hires_allowed(image, self._hires_available):
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
        log_hires_info(p)

        if not processed or not processed.images:
            print("[Menezcale] Nenhuma imagem processada encontrada.")
            return

        # Sempre guarda a última imagem gerada para o botão de carregamento.
        self._last_image = safe_copy_image(processed.images[-1])
        attach_base_metadata(
            getattr(self._last_image, "info", {}),
            original_size=None,
            p=p,
        )

        self._hires_available = is_hires_allowed(self._last_image, self._hires_available)
        if not self._hires_available:
            print("[Menezcale] Imagem sem Hires Fix detectado; controles bloqueados.")
            return
        # Automático desativado: apenas registra a última imagem e sai.
        print("[Menezcale] Downscale automático desativado. Use o botão 'Aplicar Downscale'.")

    def _load_last_image(self):
        img = getattr(self, "_last_image", None)
        if img is None:
            print("[Menezcale] Nenhuma imagem gerada anteriormente para carregar.")
            return None, None
        if not is_hires_allowed(img, self._hires_available):
            print("[Menezcale] Hires Fix não detectado na última imagem; controles bloqueados.")
            return None, None
        try:
            copy_img = img.copy()
            copy_img.info = getattr(img, "info", {}).copy()
            return copy_img, copy_img
        except Exception as err:
            print(f"[Menezcale] Falha ao carregar última imagem: {err}")
            return None, None

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

        original_size = detect_original_size(
            p=p,
            image=image,
            use_auto_original=use_auto_original,
            manual_width=manual_width,
            manual_height=manual_height,
        )

        attach_base_metadata(original_info, original_size, p)

        target_size = compute_target_size(
            image=image,
            original_size=original_size,
            down_factor=down_factor,
            use_manual_down=use_manual_down,
        )

        image = apply_downscale(image, down_method, target_size, original_info)
        image = apply_face_restore_if_enabled(image, original_info)

        print(
            f"Menezcale: Downscale para {target_size[0]}x{target_size[1]} aplicado "
            f"com {down_method} "
            f"(alvo {'original' if original_size else ('fator manual' if use_manual_down else 'tamanho atual')})"
        )

        # Reattach metadata for downstream consumers.
        image.info.update(original_info)
        return image
