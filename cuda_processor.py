import argparse
import glob
import json
import logging
import math
import os
from pathlib import Path
import platform
import time
import warnings
from datetime import datetime, timezone
from typing import Any

import numpy as np

import config


logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [CUDA] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("CUDAProcessor")


FEATURES = [
    {
        "key": "temperatura_prom",
        "global_key": "temperatura_prom_global",
        "label": "Temperatura",
        "unit": "C",
    },
    {
        "key": "humedad_prom",
        "global_key": "humedad_prom_global",
        "label": "Humedad",
        "unit": "%",
    },
    {
        "key": "presion_prom",
        "global_key": "presion_prom_global",
        "label": "Presion",
        "unit": "inHg",
    },
    {
        "key": "velocidad_viento_prom",
        "global_key": "velocidad_viento_prom_global",
        "label": "Velocidad viento",
        "unit": "mph",
    },
    {
        "key": "lluvia_prom",
        "global_key": "lluvia_prom_global",
        "label": "Lluvia",
        "unit": "in/min",
    },
    {
        "key": "nivel_luz_prom",
        "global_key": "nivel_luz_prom_global",
        "label": "Nivel luz",
        "unit": "raw",
    },
]

RISK_WEIGHTS = np.array([0.22, 0.16, 0.14, 0.22, 0.18, 0.08], dtype=np.float32)
_DLL_DIRECTORY_HANDLES = []


def _configure_windows_cuda_dll_paths() -> list[str]:
    """
    En Windows, CuPy necesita encontrar DLLs como nvrtc64_*.dll.

    A veces CUDA esta instalado, pero su carpeta bin no esta en PATH/CUDA_PATH.
    Esta funcion agrega rutas comunes al buscador de DLLs de Python antes de
    importar CuPy.
    """
    if platform.system() != "Windows":
        return []

    candidates: list[Path] = []
    cuda_path = os.getenv("CUDA_PATH")
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin")

    cuda_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if cuda_root.exists():
        candidates.extend(sorted(cuda_root.glob(r"v12*\bin"), reverse=True))
        candidates.extend(sorted(cuda_root.glob(r"v13*\bin"), reverse=True))

    added = []
    current_path = os.environ.get("PATH", "")
    path_parts = current_path.split(os.pathsep) if current_path else []

    for candidate in candidates:
        if not candidate.exists():
            continue

        has_nvrtc = any(candidate.glob("nvrtc*.dll"))
        if not has_nvrtc:
            continue

        candidate_str = str(candidate)
        if candidate_str not in path_parts:
            os.environ["PATH"] = candidate_str + os.pathsep + os.environ.get("PATH", "")

        add_dll_directory = getattr(os, "add_dll_directory", None)
        if add_dll_directory is not None:
            try:
                _DLL_DIRECTORY_HANDLES.append(add_dll_directory(candidate_str))
            except OSError:
                pass

        added.append(candidate_str)

    return added


def _latest_processed_report(prefix: str) -> str:
    pattern = os.path.join(config.PROCESSED_DIR, f"{prefix}_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No se encontro ningun archivo {prefix}_*.json en {config.PROCESSED_DIR}"
        )
    return files[-1]


def _to_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def _impute_nan(matrix: np.ndarray) -> np.ndarray:
    clean = matrix.astype(np.float32, copy=True)
    with np.errstate(invalid="ignore"):
        col_means = np.nanmean(clean, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0).astype(np.float32)
    missing_rows, missing_cols = np.where(~np.isfinite(clean))
    clean[missing_rows, missing_cols] = col_means[missing_cols]
    return clean


def _extract_feature_matrix(reporte_mpi: dict) -> tuple[np.ndarray, list[str]]:
    thingspeak = reporte_mpi.get("thingspeak", {})
    per_process = thingspeak.get("resultados_por_proceso") or []

    rows: list[list[float]] = []
    labels: list[str] = []

    for idx, item in enumerate(per_process):
        row = [_to_float(item.get(feature["key"])) for feature in FEATURES]
        if any(math.isfinite(value) for value in row):
            rows.append(row)
            labels.append(str(item.get("rank", idx)))

    if not rows:
        row = [_to_float(thingspeak.get(feature["global_key"])) for feature in FEATURES]
        if any(math.isfinite(value) for value in row):
            rows.append(row)
            labels.append("global")

    if not rows:
        raise ValueError("El reporte MPI no contiene metricas numericas para CUDA.")

    return _impute_nan(np.array(rows, dtype=np.float32)), labels


def _risk_level(score: float) -> str:
    if score >= 65:
        return "ALTO"
    if score >= 35:
        return "MEDIO"
    return "BAJO"


def _corr_cpu(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.eye(matrix.shape[1], dtype=np.float32)

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(matrix, rowvar=False).astype(np.float32)

    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _run_cpu(matrix: np.ndarray, reason: str) -> dict:
    start = time.perf_counter()
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    safe_stds = np.where(stds > 1e-6, stds, 1.0).astype(np.float32)
    zscores = (matrix - means) / safe_stds
    risk = np.clip(np.abs(zscores).dot(RISK_WEIGHTS) * 30.0, 0.0, 100.0)
    corr = _corr_cpu(matrix)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "backend": "cpu_fallback",
        "device": "CPU",
        "fallback_reason": reason,
        "elapsed_ms": round(elapsed_ms, 3),
        "means": means,
        "stds": stds,
        "mins": matrix.min(axis=0),
        "maxs": matrix.max(axis=0),
        "zscores": zscores,
        "risk": risk,
        "corr": corr,
    }


def _run_cuda(matrix: np.ndarray) -> dict:
    try:
        added_paths = _configure_windows_cuda_dll_paths()
        if added_paths:
            logger.info("Rutas CUDA agregadas para DLLs: %s", added_paths)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="CUDA path could not be detected.*",
                category=UserWarning,
            )
            import cupy as cp

        if cp.cuda.runtime.getDeviceCount() == 0:
            return _run_cpu(matrix, "CuPy instalado, pero no hay GPU CUDA disponible.")
    except Exception as exc:
        return _run_cpu(matrix, _cuda_error_hint(exc))

    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        gpu_name = props.get("name", b"GPU")
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode(errors="replace")

        risk_kernel = cp.ElementwiseKernel(
            "float32 temp_z, float32 hum_z, float32 pres_z, "
            "float32 wind_z, float32 rain_z, float32 light_z",
            "float32 score",
            """
            float acc = 0.0f;
            acc += 0.22f * fabsf(temp_z);
            acc += 0.16f * fabsf(hum_z);
            acc += 0.14f * fabsf(pres_z);
            acc += 0.22f * fabsf(wind_z);
            acc += 0.18f * fabsf(rain_z);
            acc += 0.08f * fabsf(light_z);
            score = fminf(100.0f, acc * 30.0f);
            """,
            "environmental_risk_kernel",
        )

        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        start_event.record()

        gpu_matrix = cp.asarray(matrix, dtype=cp.float32)
        means = cp.mean(gpu_matrix, axis=0)
        stds = cp.std(gpu_matrix, axis=0)
        safe_stds = cp.where(stds > 1e-6, stds, 1.0)
        zscores = (gpu_matrix - means) / safe_stds
        risk = risk_kernel(
            zscores[:, 0],
            zscores[:, 1],
            zscores[:, 2],
            zscores[:, 3],
            zscores[:, 4],
            zscores[:, 5],
        )

        if matrix.shape[0] < 2:
            corr = cp.eye(matrix.shape[1], dtype=cp.float32)
        else:
            corr = cp.corrcoef(gpu_matrix, rowvar=False)
            corr = cp.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            diag = cp.arange(matrix.shape[1])
            corr[diag, diag] = 1.0

        mins = cp.min(gpu_matrix, axis=0)
        maxs = cp.max(gpu_matrix, axis=0)

        end_event.record()
        end_event.synchronize()
        elapsed_ms = cp.cuda.get_elapsed_time(start_event, end_event)

        return {
            "backend": "cuda",
            "device": gpu_name,
            "fallback_reason": None,
            "elapsed_ms": round(float(elapsed_ms), 3),
            "means": cp.asnumpy(means),
            "stds": cp.asnumpy(stds),
            "mins": cp.asnumpy(mins),
            "maxs": cp.asnumpy(maxs),
            "zscores": cp.asnumpy(zscores),
            "risk": cp.asnumpy(risk),
            "corr": cp.asnumpy(corr),
        }
    except Exception as exc:
        logger.warning("CUDA fallo durante la ejecucion; se usa fallback CPU: %s", exc)
        return _run_cpu(matrix, _cuda_error_hint(exc, during_execution=True))


def _cuda_error_hint(exc: Exception, during_execution: bool = False) -> str:
    message = str(exc)
    prefix = "CUDA fallo durante la ejecucion" if during_execution else "CUDA/CuPy no disponible"

    if "nvrtc" in message.lower() and "dll" in message.lower():
        return (
            f"{prefix}: {message}. Falta NVRTC en el PATH de Windows. "
            "Solucion recomendada: instala/actualiza con "
            "pip install -U \"cupy-cuda12x[ctk]\" o agrega "
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x\\bin "
            "a CUDA_PATH/PATH."
        )

    return f"{prefix}: {message}"


def _round_list(values: np.ndarray, digits: int = 4) -> list[float]:
    return [round(float(value), digits) for value in values.tolist()]


def _round_matrix(values: np.ndarray, digits: int = 4) -> list[list[float]]:
    return [[round(float(value), digits) for value in row] for row in values.tolist()]


def _stats_by_feature(result: dict) -> dict:
    stats = {}
    for idx, feature in enumerate(FEATURES):
        stats[feature["key"]] = {
            "label": feature["label"],
            "unit": feature["unit"],
            "mean": round(float(result["means"][idx]), 4),
            "std": round(float(result["stds"][idx]), 4),
            "min": round(float(result["mins"][idx]), 4),
            "max": round(float(result["maxs"][idx]), 4),
        }
    return stats


def _risk_rows(result: dict, labels: list[str]) -> list[dict]:
    rows = []
    for idx, score in enumerate(result["risk"]):
        numeric_score = round(float(score), 2)
        rows.append(
            {
                "rank": labels[idx],
                "score": numeric_score,
                "nivel": _risk_level(numeric_score),
                "zscores": {
                    FEATURES[col]["key"]: round(float(result["zscores"][idx][col]), 4)
                    for col in range(len(FEATURES))
                },
            }
        )
    return rows


def _interpret(result: dict, risk_rows: list[dict]) -> list[str]:
    messages = []
    backend = result["backend"]
    if backend == "cuda":
        messages.append(
            "La fase 3 uso CuPy sobre una GPU NVIDIA para normalizar metricas, "
            "calcular correlaciones y ejecutar un kernel de criticidad relativa."
        )
    else:
        messages.append(
            "La fase 3 se ejecuto en CPU porque CUDA no estuvo disponible; el "
            "codigo conserva la misma ruta de calculo para poder validarse sin GPU."
        )

    if risk_rows:
        top = max(risk_rows, key=lambda row: row["score"])
        messages.append(
            f"El bloque/rank con mayor criticidad relativa fue {top['rank']} "
            f"con score {top['score']} ({top['nivel']})."
        )

    corr = result["corr"]
    pairs = []
    for i in range(len(FEATURES)):
        for j in range(i + 1, len(FEATURES)):
            value = float(corr[i][j])
            if abs(value) >= 0.7:
                pairs.append((abs(value), value, FEATURES[i]["label"], FEATURES[j]["label"]))

    if pairs:
        pairs.sort(reverse=True)
        _, value, left, right = pairs[0]
        messages.append(
            f"La correlacion mas fuerte detectada fue {left} vs {right} "
            f"con r={value:.2f}."
        )
    else:
        messages.append(
            "No se detectaron correlaciones fuertes entre variables con los datos disponibles."
        )

    return messages


def run_cuda_analysis(input_report: str | None = None) -> str:
    input_path = input_report or _latest_processed_report("reporte_mpi")
    with open(input_path, encoding="utf-8") as file:
        reporte_mpi = json.load(file)

    matrix, labels = _extract_feature_matrix(reporte_mpi)
    logger.info("Matriz de entrada para CUDA: %s bloques x %s variables", *matrix.shape)

    result = _run_cuda(matrix)
    risk_rows = _risk_rows(result, labels)
    interpretation = _interpret(result, risk_rows)

    cuda_report = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_mpi_report": os.path.basename(input_path),
            "backend": result["backend"],
            "device": result["device"],
            "fallback_reason": result["fallback_reason"],
            "elapsed_ms": result["elapsed_ms"],
            "n_bloques": int(matrix.shape[0]),
            "n_variables": int(matrix.shape[1]),
        },
        "variables": FEATURES,
        "estadisticas": _stats_by_feature(result),
        "correlacion": {
            "labels": [feature["label"] for feature in FEATURES],
            "matrix": _round_matrix(result["corr"]),
        },
        "riesgo_relativo_por_bloque": risk_rows,
        "matriz_normalizada_zscore": _round_matrix(result["zscores"]),
        "interpretacion": interpretation,
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(config.PROCESSED_DIR, f"reporte_cuda_{ts}.json")
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(cuda_report, file, ensure_ascii=False, indent=2)

    logger.info(
        "Reporte CUDA guardado en %s usando backend %s",
        output_path,
        result["backend"],
    )

    print("\n" + "=" * 60)
    print("  RESUMEN DEL ANALISIS CUDA")
    print("=" * 60)
    print(f"  Backend              : {result['backend']}")
    print(f"  Dispositivo          : {result['device']}")
    print(f"  Tiempo calculo       : {result['elapsed_ms']} ms")
    print(f"  Bloques analizados   : {matrix.shape[0]}")
    print(f"  Variables analizadas : {matrix.shape[1]}")
    if result["fallback_reason"]:
        print(f"  Nota                 : {result['fallback_reason']}")
    print(f"  Reporte              : {output_path}")
    print("=" * 60 + "\n")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecuta la fase CUDA del proyecto.")
    parser.add_argument(
        "--input",
        help="Ruta opcional a un reporte_mpi_*.json. Si se omite, usa el mas reciente.",
    )
    args = parser.parse_args()
    run_cuda_analysis(args.input)


if __name__ == "__main__":
    main()
