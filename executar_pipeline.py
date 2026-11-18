import subprocess
import sys
from pathlib import Path


ETAPAS = (
    "01_preparar_prf.py",
    "02_preparar_inmet.py",
    "03_integrar_bases.py",
    "04_analise_descritiva.py",
    "05_regressao_ordinal.py",
)


def main() -> int:
    pasta_src = Path(__file__).resolve().parent / "src"

    for etapa in ETAPAS:
        print(f"Executando {etapa}", flush=True)
        resultado = subprocess.run([sys.executable, str(pasta_src / etapa)])
        if resultado.returncode != 0:
            return resultado.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
