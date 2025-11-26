"""
Instala dependências leves opcionais para a extensão Menezcale.
Atualmente apenas sd-parsers é necessário para parsear metadados SD
quando disponível. A instalação é resiliente para ambientes offline.
"""

import subprocess
import sys


def ensure_package(pkg: str):
    try:
        __import__(pkg.replace("-", "_"))
        print(f"[Menezcale] Dependência '{pkg}' já instalada.")
        return
    except Exception:
        pass

    try:
        print(f"[Menezcale] Instalando '{pkg}' via pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        print(f"[Menezcale] '{pkg}' instalada com sucesso.")
    except Exception as err:
        print(f"[Menezcale] Não foi possível instalar '{pkg}': {err}")


def main():
    ensure_package("sd-parsers")


if __name__ == "__main__":
    main()
