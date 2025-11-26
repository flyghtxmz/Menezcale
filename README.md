# Menezcale - Auto Downscale (Forge / SD-WebUI)

Extensão para o painel **txt2img** do Forge (fork do Stable Diffusion WebUI) focada em recuperar nitidez depois de um upscale externo ou opcional pré-upscale automático. O fluxo detecta o tamanho original, aplica (opcionalmente) um upscale rápido e então faz downscale inteligente para voltar ao tamanho original preservando detalhes.

## Instalação

1. Coloque esta pasta em `extensions/menezcale/`.
2. (Opcional) Rode `python install.py` dentro da pasta para instalar `sd-parsers` (usado para ler metadados do PNG).
3. Baixe modelos que desejar usar, por exemplo `FSRCNN_x2.pth`, em `models/ESRGAN/`.
4. Recarregue a UI do Forge / SD-WebUI.

## Uso

- Abra a aba **txt2img** e, no painel direito de scripts, abra o accordion **"Menezcale - Auto Downscale (Foco em Nitidez Pós-Upscale)"**.
- Itens principais:
  - **Ativar Menezcale**: liga/desliga o fluxo pós-geração (ativo por padrão).
  - **Incluir Upscale Automático Antes**: se marcado, aplica um upscale rápido antes do downscale usando o modelo e fator escolhidos.
  - **Método de Downscale**: `Lanczos` (mais fiel), `FSRCNN` (se o modelo existir em `models/ESRGAN`) ou `Bicubic`.
  - **Fator de Downscale**: razão aplicada sobre a imagem atual (0.25 volta ao original após um upscale 4x).
  - **Usar Tamanho Original Automático**: detecta `p.width/p.height` ou metadados do PNG; ao desligar, habilita sliders para largura/altura manual.
  - **Teste Manual**: envie uma imagem (pós-upscale) e clique em **Testar Downscale Manual** para ver o preview imediato.
- O hook roda no `postprocess` do txt2img, iterando sobre todas as imagens em `processed.images`.

## Como funciona

1. Detecta o tamanho original via `p.width/p.height`; fallback por regex ou `sd-parsers` nos metadados `parameters` do PNG; se desativar o modo automático, usa os sliders de tamanho manual.
2. (Opcional) Upscale automático com os upscalers carregados (`4x-UltraSharp`, `R-ESRGAN 4x+` ou `None`), usando o fator configurado.
3. Downscale para o tamanho alvo:
   - **Lanczos**: `PIL.Image.resize` com `Image.Resampling.LANCZOS`.
   - **Bicubic**: `PIL.Image.BICUBIC`.
   - **FSRCNN**: tenta usar um upscaler que contenha `FSRCNN` (modelo `FSRCNN_x2.pth`), ajustando para o alvo; fallback é Lanczos.
4. Copia metadados de volta para a imagem final e loga no console o método e tamanho aplicados.
5. Se o GFPGAN estiver habilitado no WebUI (`face_restoration_model`), aplica polimento de faces no resultado final.

## Dica de teste rápido

- Gere uma imagem 512x512 em txt2img, marque **Incluir Upscale Automático** com fator 4x (ou faça upscale externo 4x), mantenha **Fator de Downscale** em `0.25` e **Usar Tamanho Original Automático** ligado. A imagem final deve voltar a ~512x512 com nitidez preservada.

## Pastas adicionais

- `scripts/menezcale_script.py`: lógica da extensão e UI Gradio.
- `install.py`: instala `sd-parsers` (opcional).

Logs de debug com prefixo `[Menezcale]` são emitidos no console do WebUI para facilitar verificação de tamanho/método aplicado.
