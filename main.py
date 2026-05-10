import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import asyncio
import subprocess
import logging
import os
import time

import config
from data_fetcher import fetch_all_sources

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [MAIN] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("Main")


def fase_1_fetch() -> dict:
    """
    Fase 1: Lectura concurrente de todas las APIs con Asyncio.

    Llama a fetch_all_sources() que internamente usa asyncio.gather
    para lanzar todas las peticiones HTTP simultáneamente.
    """
    logger.info("▶ FASE 1: Iniciando fetching concurrente...")
    t0 = time.perf_counter()

    # asyncio.run() es el punto de entrada al mundo asíncrono.
    # Crea el event loop, ejecuta la corrutina hasta completarse y lo cierra.
    resultados = asyncio.run(fetch_all_sources())

    elapsed = time.perf_counter() - t0
    logger.info(f"✓ FASE 1 completada en {elapsed:.2f}s")

    # Resumen rápido de lo obtenido
    for key, val in resultados.items():
        if hasattr(val, "__len__") and key != "timestamp":
            logger.info(f"  {key}: {len(val)} registros")

    return resultados


def fase_2_mpi(n_procesos: int = 4) -> bool:
    """
    Fase 2: Procesamiento distribuido con MPI4Py.

    Lanza mpiexec con N procesos. Cada proceso corre mpi_processor.py,
    lee su porción de datos desde disco y ejecuta los análisis en paralelo.

    Parámetros
    ----------
    n_procesos : int
        Número de procesos MPI a lanzar. Recomendado: igual al número
        de núcleos físicos de tu CPU (o menos).
    """
    logger.info(f"▶ FASE 2: Iniciando procesamiento MPI con {n_procesos} procesos...")
    t0 = time.perf_counter()

    # Construir el comando mpiexec
    # -n N : número de procesos
    # sys.executable : usa el mismo Python del entorno virtual actual
    cmd = [
        "mpiexec",
        "-n", str(n_procesos),
        sys.executable,           # python / python3 del entorno actual
        os.path.join(os.path.dirname(__file__), "mpi_processor.py"),
    ]

    logger.info(f"Comando: {' '.join(cmd)}")

    try:
        # subprocess.run bloquea hasta que mpiexec termine.
        # check=True lanza CalledProcessError si el proceso falla.
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,   # mostrar output en consola directamente
        )
        elapsed = time.perf_counter() - t0
        logger.info(f"✓ FASE 2 completada en {elapsed:.2f}s (exit code {result.returncode})")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"✗ MPI falló con código {e.returncode}")
        return False
    except FileNotFoundError:
        logger.error(
            "✗ 'mpiexec' no encontrado. "
            "Instala MPI: sudo apt install libopenmpi-dev  |  brew install open-mpi"
        )
        return False


def main():
    """
    Punto de entrada principal del pipeline.

    Argumentos de línea de comandos opcionales:
        python main.py [n_procesos]
        Ejemplo: python main.py 8   → usa 8 procesos MPI
    """
    print("\n" + "═" * 60)
    print("  PIPELINE: ANÁLISIS DE TRÁFICO AÉREO Y SENSORES AMBIENTALES")
    print("═" * 60)

    # Número de procesos MPI (por defecto 4; se puede sobreescribir por CLI)
    n_procesos = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    #Asyncio
    try:
        datos = fase_1_fetch()
    except Exception as e:
        logger.error(f"FASE 1 falló: {e}")
        sys.exit(1)

    #FASE 2: MPI
    exito = fase_2_mpi(n_procesos)
    if not exito:
        logger.error("FASE 2 falló. Revisa los logs.")
        sys.exit(1)

    print("\n" + "═" * 60)
    print("  Pipeline completado. Los archivos de salida están en:")
    print(f"    Datos crudos     → {config.RAW_DATA_DIR}")
    print(f"      • thingspeak_*.csv   : series temporales de sensores")
    print(f"      • weather_*.json     : condiciones meteorologicas")
    print(f"    Datos procesados → {config.PROCESSED_DIR}")
    print(f"      • reporte_mpi_*.json : analisis agregado listo para CUDA")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()