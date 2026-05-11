import glob
import io
import json
import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from mpi4py import MPI

import config


if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
SIZE = comm.Get_size()

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=f"%(asctime)s [RANK {RANK}/{SIZE}] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("MPIProcessor")


def _load_latest_csv(prefix: str) -> pd.DataFrame:
    """Carga el CSV mas reciente cuyo nombre empieza con el prefijo indicado."""
    pattern = os.path.join(config.RAW_DATA_DIR, f"{prefix}_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        logger.warning("No se encontraron archivos para prefijo '%s'", prefix)
        return pd.DataFrame()

    latest = files[-1]
    logger.info("Cargando %s", latest)
    return pd.read_csv(latest)


def _load_latest_json(prefix: str) -> dict:
    """Carga el JSON mas reciente cuyo nombre empieza con el prefijo indicado."""
    pattern = os.path.join(config.RAW_DATA_DIR, f"{prefix}_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return {}

    with open(files[-1], encoding="utf-8") as file:
        return json.load(file)


def _series_stats(serie: pd.Series) -> dict:
    q1, q3 = serie.quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = serie[(serie < q1 - 1.5 * iqr) | (serie > q3 + 1.5 * iqr)]

    return {
        "prom": float(serie.mean()),
        "max": float(serie.max()),
        "min": float(serie.min()),
        "std": float(serie.std(ddof=0)),
        "outliers": int(len(outliers)),
    }


def analizar_thingspeak(df: pd.DataFrame) -> dict:
    """
    Calcula estadistica descriptiva y outliers IQR para el fragmento asignado.
    """
    if df.empty:
        return {"rank": RANK, "error": "DataFrame vacio"}

    resultado = {"rank": RANK, "n_registros": len(df)}
    variables = [
        "temperatura",
        "humedad",
        "presion",
        "lluvia",
        "nivel_voltaje",
        "nivel_luz",
        "velocidad_viento",
        "direccion_viento",
    ]

    for var in variables:
        if var not in df.columns:
            continue

        serie = pd.to_numeric(df[var], errors="coerce").dropna()
        if serie.empty:
            continue

        if var == "temperatura":
            serie = (serie - 32) * 5 / 9

        stats = _series_stats(serie)
        resultado[f"{var}_prom"] = stats["prom"]
        resultado[f"{var}_max"] = stats["max"]
        resultado[f"{var}_min"] = stats["min"]
        resultado[f"{var}_std"] = stats["std"]
        resultado[f"{var}_outliers"] = stats["outliers"]

    if "created_at" in df.columns and "temperatura" in df.columns:
        df_t = df[["created_at", "temperatura"]].copy()
        df_t["created_at"] = pd.to_datetime(df_t["created_at"], errors="coerce")
        df_t["temperatura"] = pd.to_numeric(df_t["temperatura"], errors="coerce")
        df_t = df_t.dropna()

        if len(df_t) >= 2:
            t0 = df_t["created_at"].min()
            x = (df_t["created_at"] - t0).dt.total_seconds()
            y = (df_t["temperatura"] - 32) * 5 / 9
            coef = float(np.polyfit(x, y, 1)[0])
            resultado["temperatura_tendencia"] = (
                "subiendo"
                if coef > 0.001
                else "bajando"
                if coef < -0.001
                else "estable"
            )

    return resultado


def analizar_openweather(clima: dict) -> dict:
    """Resume las condiciones meteorologicas y calcula adversidad ambiental."""
    if not clima:
        return {"error": "Sin datos de OpenWeather"}

    resultado = {
        "ciudad": clima.get("ciudad"),
        "descripcion": clima.get("descripcion"),
        "temperatura_c": clima.get("temperatura"),
        "humedad_pct": clima.get("humedad"),
        "presion_hpa": clima.get("presion"),
        "visibilidad_m": clima.get("visibilidad"),
        "viento_kmh": clima.get("velocidad_viento"),
        "nubosidad_pct": clima.get("nubosidad"),
    }

    visib = float(clima.get("visibilidad", 10000) or 10000)
    viento = float(clima.get("velocidad_viento", 0) or 0)
    nubos = float(clima.get("nubosidad", 0) or 0)

    adversidad = min(
        100,
        (1 - min(visib, 10000) / 10000) * 40
        + min(viento / 30, 1) * 35
        + (nubos / 100) * 25,
    )

    resultado["adversidad_ambiental_idx"] = round(adversidad, 2)
    resultado["nivel_adversidad"] = (
        "ALTO" if adversidad > 60 else "MEDIO" if adversidad > 30 else "BAJO"
    )
    return resultado


def correlacion_sensores_clima(
    resultados_thingspeak: list,
    resultado_clima: dict,
) -> dict:
    """Cruza el resumen de ThingSpeak con OpenWeather cuando hay datos."""
    correlacion = {}

    temps_ts = [
        r.get("temperatura_prom")
        for r in resultados_thingspeak
        if r.get("temperatura_prom") is not None
    ]
    if temps_ts and resultado_clima.get("temperatura_c") is not None:
        prom_ts = float(np.mean(temps_ts))
        prom_ow = float(resultado_clima["temperatura_c"])
        diferencia = abs(prom_ts - prom_ow)
        correlacion["temperatura_thingspeak_prom"] = round(prom_ts, 2)
        correlacion["temperatura_openweather"] = prom_ow
        correlacion["diferencia_temperatura_c"] = round(diferencia, 2)
        correlacion["concordancia_temperatura"] = (
            "BUENA" if diferencia < 2 else "MODERADA" if diferencia < 5 else "BAJA"
        )

    hums_ts = [
        r.get("humedad_prom")
        for r in resultados_thingspeak
        if r.get("humedad_prom") is not None
    ]
    if hums_ts and resultado_clima.get("humedad_pct") is not None:
        correlacion["humedad_thingspeak_prom"] = round(float(np.mean(hums_ts)), 2)
        correlacion["humedad_openweather"] = resultado_clima["humedad_pct"]

    adversidad = resultado_clima.get("adversidad_ambiental_idx", 0)
    correlacion["interpretacion_ambiental"] = (
        "Alta adversidad climatica: revisar condiciones de visibilidad y viento"
        if adversidad > 60
        else "Condiciones ambientales dentro de rangos normales"
    )

    return correlacion


def _mean_metric(results: list[dict], key: str):
    vals = [r[key] for r in results if key in r and r[key] is not None]
    return round(float(np.mean(vals)), 2) if vals else None


def run_mpi_pipeline():
    """
    Orquesta el procesamiento distribuido.

    Rank 0 carga datos y divide registros. Todos los ranks analizan su bloque y
    rank 0 agrega resultados en datos_procesados/reporte_mpi_*.json.
    """
    if RANK == 0:
        logger.info("=" * 60)
        logger.info("MPI Pipeline iniciado con %s proceso(s)", SIZE)
        logger.info("=" * 60)

        df_thingspeak = _load_latest_csv("thingspeak")
        clima = _load_latest_json("weather")

        logger.info("ThingSpeak cargado  : %s registros", len(df_thingspeak))
        logger.info("OpenWeather cargado : %s campos", len(clima))

        if not df_thingspeak.empty:
            indices = np.array_split(np.arange(len(df_thingspeak)), SIZE)
            chunks_ts = [
                df_thingspeak.iloc[idx].to_dict(orient="records")
                for idx in indices
            ]
        else:
            chunks_ts = [[] for _ in range(SIZE)]
    else:
        chunks_ts = None
        clima = None

    registros_recibidos = comm.scatter(chunks_ts, root=0)
    mi_chunk_ts = pd.DataFrame(registros_recibidos)
    clima = comm.bcast(clima, root=0)

    comm.Barrier()
    logger.info("Datos recibidos: %s registros ThingSpeak", len(mi_chunk_ts))

    logger.info("Iniciando analisis local...")
    resultado_ts = analizar_thingspeak(mi_chunk_ts)
    logger.info("Analisis local completado.")

    todos_ts = comm.gather(resultado_ts, root=0)

    if RANK != 0:
        return None

    logger.info("Agregando resultados de todos los procesos...")
    todos_ts_ok = [r for r in todos_ts if "error" not in r]

    resultado_clima = analizar_openweather(clima)
    correlacion = correlacion_sensores_clima(todos_ts_ok, resultado_clima)

    variables = [
        "temperatura",
        "humedad",
        "presion",
        "lluvia",
        "nivel_voltaje",
        "nivel_luz",
        "velocidad_viento",
        "direccion_viento",
    ]
    outliers_global = {
        var: sum(r.get(f"{var}_outliers", 0) for r in todos_ts_ok)
        for var in variables
    }

    reporte = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "procesos_mpi": SIZE,
            "total_registros_thingspeak": sum(
                r.get("n_registros", 0) for r in todos_ts_ok
            ),
        },
        "thingspeak": {
            "temperatura_prom_global": _mean_metric(todos_ts_ok, "temperatura_prom"),
            "humedad_prom_global": _mean_metric(todos_ts_ok, "humedad_prom"),
            "presion_prom_global": _mean_metric(todos_ts_ok, "presion_prom"),
            "velocidad_viento_prom_global": _mean_metric(
                todos_ts_ok, "velocidad_viento_prom"
            ),
            "direccion_viento_prom_global": _mean_metric(
                todos_ts_ok, "direccion_viento_prom"
            ),
            "lluvia_prom_global": _mean_metric(todos_ts_ok, "lluvia_prom"),
            "nivel_luz_prom_global": _mean_metric(todos_ts_ok, "nivel_luz_prom"),
            "alerta_pm25": "NORMAL",
            "outliers_detectados": outliers_global,
            "tendencias_temperatura": [
                r.get("temperatura_tendencia")
                for r in todos_ts_ok
                if r.get("temperatura_tendencia")
            ],
            "resultados_por_proceso": todos_ts_ok,
        },
        "openweather": resultado_clima,
        "correlacion_fuentes": correlacion,
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    reporte_path = os.path.join(config.PROCESSED_DIR, f"reporte_mpi_{ts}.json")
    with open(reporte_path, "w", encoding="utf-8") as file:
        json.dump(reporte, file, ensure_ascii=False, indent=2, default=str)

    logger.info("Reporte guardado en: %s", reporte_path)

    print("\n" + "=" * 60)
    print("  RESUMEN DEL ANALISIS MPI - DATOS AMBIENTALES")
    print("=" * 60)
    print(f"  Procesos MPI                : {SIZE}")
    print(f"  Registros ThingSpeak        : {reporte['metadata']['total_registros_thingspeak']}")
    print(f"  Temperatura prom. (TS)      : {reporte['thingspeak']['temperatura_prom_global']} C")
    print(f"  Temperatura (OpenWeather)   : {resultado_clima.get('temperatura_c')} C")
    print(f"  Concordancia temp.          : {correlacion.get('concordancia_temperatura', 'N/A')}")
    print(f"  Humedad prom. (TS)          : {reporte['thingspeak']['humedad_prom_global']} %")
    print(
        "  Velocidad viento prom. (TS) : "
        f"{reporte['thingspeak']['velocidad_viento_prom_global']} mph"
    )
    print(f"  Lluvia prom. (TS)           : {reporte['thingspeak']['lluvia_prom_global']} in/min")
    print(f"  Nivel luz prom. (TS)        : {reporte['thingspeak']['nivel_luz_prom_global']}")
    print(f"  Adversidad ambiental        : {resultado_clima.get('adversidad_ambiental_idx')}/100")
    print(f"  Nivel adversidad            : {resultado_clima.get('nivel_adversidad')}")
    print(f"  Interpretacion              : {correlacion.get('interpretacion_ambiental')}")
    print(f"  Outliers temperatura        : {outliers_global.get('temperatura', 0)}")
    print(f"  Outliers velocidad viento   : {outliers_global.get('velocidad_viento', 0)}")
    print("=" * 60)
    print(f"  Reporte -> {reporte_path}")
    print("=" * 60 + "\n")

    return reporte


if __name__ == "__main__":
    if RANK == 0:
        raw_files = glob.glob(os.path.join(config.RAW_DATA_DIR, "*.csv"))
        if not raw_files:
            logger.error(
                "No se encontraron datos crudos. Ejecuta primero: python data_fetcher.py"
            )
            comm.Abort(1)
            sys.exit(1)

    run_mpi_pipeline()
    MPI.Finalize()
