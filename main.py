import asyncio
import io
import logging
import os
import subprocess
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import config
from cuda_processor import run_cuda_analysis
from data_fetcher import fetch_all_sources
from report_generator import generate_html_report


logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
    ],
    force=True,
)
logger = logging.getLogger("Main")


def fase_1_fetch() -> dict:
    """
    Fase 1: lectura concurrente de las APIs con Asyncio.

    fetch_all_sources() usa asyncio.gather para lanzar las peticiones HTTP al
    mismo tiempo y guardar los datos crudos en disco para que MPI pueda leerlos.
    """
    logger.info("> FASE 1: Iniciando fetching concurrente...")
    t0 = time.perf_counter()

    resultados = asyncio.run(fetch_all_sources())

    elapsed = time.perf_counter() - t0
    logger.info("OK FASE 1 completada en %.2fs", elapsed)

    for key, val in resultados.items():
        if hasattr(val, "__len__") and key != "timestamp" and val is not None:
            logger.info("  %s: %s registros", key, len(val))

    return resultados


def fase_2_mpi(n_procesos: int = 4) -> bool:
    """
    Fase 2: procesamiento distribuido con MPI4Py.

    Lanza mpiexec con N procesos. Cada proceso corre mpi_processor.py, recibe
    una parte del dataset, calcula estadisticas locales y el rank 0 agrega el
    reporte final en datos_procesados/reporte_mpi_*.json.
    """
    logger.info("> FASE 2: Iniciando MPI con %s procesos...", n_procesos)
    t0 = time.perf_counter()

    cmd = [
        "mpiexec",
        "-n",
        str(n_procesos),
        sys.executable,
        os.path.join(os.path.dirname(__file__), "mpi_processor.py"),
    ]

    logger.info("Comando: %s", " ".join(cmd))

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        elapsed = time.perf_counter() - t0
        logger.info(
            "OK FASE 2 completada en %.2fs (exit code %s)",
            elapsed,
            result.returncode,
        )
        return True

    except subprocess.CalledProcessError as exc:
        logger.error("MPI fallo con codigo %s", exc.returncode)
        return False
    except FileNotFoundError:
        logger.error(
            "'mpiexec' no fue encontrado. En Windows instala Microsoft MPI; "
            "en Linux instala OpenMPI o MPICH."
        )
        return False


def fase_3_cuda() -> str:
    """
    Fase 3: procesamiento matricial con CUDA/CuPy.

    Toma el reporte generado por MPI, arma una matriz numerica con las metricas
    por rank y ejecuta normalizacion, correlacion y un kernel de criticidad
    relativa. Si CUDA no esta disponible, usa fallback CPU documentado.
    """
    logger.info("> FASE 3: Iniciando analisis CUDA...")
    t0 = time.perf_counter()

    reporte_cuda = run_cuda_analysis()

    elapsed = time.perf_counter() - t0
    logger.info("OK FASE 3 completada en %.2fs", elapsed)
    return reporte_cuda


def fase_4_html() -> str:
    """
    Fase 4: generacion del reporte HTML final.

    Une las salidas de MPI y CUDA en una pagina autocontenida lista para abrir
    en el navegador y usar como evidencia del proyecto.
    """
    logger.info("> FASE 4: Generando reporte HTML...")
    t0 = time.perf_counter()

    reporte_html = generate_html_report()

    elapsed = time.perf_counter() - t0
    logger.info("OK FASE 4 completada en %.2fs", elapsed)
    return reporte_html


def main() -> None:
    """
    Punto de entrada principal del pipeline.

    Uso:
        python main.py [n_procesos]
        py main.py [n_procesos]
    """
    print("\n" + "=" * 60, flush=True)
    print("  PIPELINE: ANALISIS AMBIENTAL PARALELO Y DISTRIBUIDO", flush=True)
    print("=" * 60, flush=True)

    n_procesos = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    try:
        fase_1_fetch()
    except Exception as exc:
        logger.error("FASE 1 fallo: %s", exc)
        sys.exit(1)

    if not fase_2_mpi(n_procesos):
        logger.error("FASE 2 fallo. Revisa los logs.")
        sys.exit(1)

    try:
        fase_3_cuda()
    except Exception as exc:
        logger.error("FASE 3 fallo: %s", exc)
        sys.exit(1)

    try:
        reporte_html = fase_4_html()
    except Exception as exc:
        logger.error("FASE 4 fallo: %s", exc)
        sys.exit(1)

    print("\n" + "=" * 60, flush=True)
    print("  Pipeline completado. Archivos de salida:", flush=True)
    print(f"    Datos crudos      -> {config.RAW_DATA_DIR}")
    print("      - thingspeak_*.csv   : series temporales de sensores")
    print("      - weather_*.json     : condiciones meteorologicas")
    print(f"    Datos procesados  -> {config.PROCESSED_DIR}")
    print("      - reporte_mpi_*.json : analisis agregado MPI")
    print("      - reporte_cuda_*.json: analisis matricial CUDA")
    print(f"    Reporte HTML      -> {reporte_html}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
