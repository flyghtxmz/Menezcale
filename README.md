# Menezcale - Auto Downscale (Forge / SD-WebUI)

Extensão para o painel **txt2img** do Forge (fork do Stable Diffusion WebUI) focada em recuperar nitidez depois de um upscale externo. O fluxo detecta o tamanho original (inclusive antes do Hires Fix) e faz downscale inteligente para voltar ao tamanho original preservando detalhes. O acionamento é manual (botão) e exige Hires Fix ativo para liberar o downscale.

## Instalação

1. Coloque esta pasta em `extensions/Menezcale/` no diretório do Forge/SD-WebUI.
2. Dependência: `sd-parsers` (listada em `requirements.txt`). O script tenta instalar automaticamente via pip; se preferir, instale manualmente com `pip install -r requirements.txt`. Se a instalação falhar, o regex de fallback continua funcionando.
3. Baixe modelos que desejar usar, por exemplo `FSRCNN_x2.pth`, em `models/ESRGAN/`.
4. Recarregue a UI do Forge / SD-WebUI.

## Uso

- Abra a aba **txt2img** e, no painel direito de scripts, abra o accordion **"Menezcale - Auto Downscale (Foco em Nitidez Pós-Upscale)"**.
- Fluxo manual (recomendado):
  - Clique em **Carregar última imagem gerada** para trazer a última saída do txt2img (a imagem é exibida no preview menor).
  - Clique em **Aplicar Downscale** para voltar a imagem ao tamanho original detectado.
  - Checkbox **Tamanho original**: ligado volta para o tamanho base (p.width/p.height, Hires Fix ou metadados). Desligado habilita sliders de largura/altura manual.
  - Checkbox **Usar fator manual de downscale**: opcional; habilita o slider de fator manual em vez de usar o tamanho original.
  - **Método de Downscale**: `Lanczos` (mais fiel), `FSRCNN` (se o modelo existir em `models/ESRGAN`) ou `Bicubic`.
- Pré-visualização: o upload/preview são menores para facilitar a inspeção rápida dentro do painel.
- Importante: se o Hires Fix não estiver ativo na geração, os controles de downscale ficam bloqueados (tanto no automático quanto no manual).

## Como funciona

1. Detecta o tamanho original via `p.width/p.height`; fallback por regex ou `sd-parsers` nos metadados `parameters` do PNG (incluindo `Size:`); ou sliders manuais se desligar “Tamanho original”.
2. Downscale para o tamanho alvo:
   - **Lanczos**: `PIL.Image.resize` com `Image.Resampling.LANCZOS`.
   - **Bicubic**: `PIL.Image.BICUBIC`.
   - **FSRCNN**: tenta usar um upscaler que contenha `FSRCNN` (modelo `FSRCNN_x2.pth`), ajustando para o alvo; fallback é Lanczos.
3. Copia metadados de volta para a imagem final e loga no console o método e tamanho aplicados.
4. Se o GFPGAN estiver habilitado no WebUI (`face_restoration_model`), aplica polimento de faces no resultado final.

## Pastas adicionais

- `scripts/menezcale_script.py`: lógica da extensão e UI Gradio.
- `scripts/menezcale_core.py`: helpers de detecção de tamanho, downscale, metadados e GFPGAN.
- `install.py`: redundante (instala `sd-parsers` manualmente se desejar), já que o script tenta resolver automaticamente.
- `requirements.txt`: lista `sd-parsers`.

Logs de debug com prefixo `[Menezcale]` são emitidos no console do WebUI para facilitar verificação de tamanho/método aplicado.
