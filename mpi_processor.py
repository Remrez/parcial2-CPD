import sys
import io

# Forzar UTF-8 en stdout y stderr para compatibilidad con Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import glob
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from mpi4py import MPI

import config

#Configuración del logger
comm = MPI.COMM_WORLD
RANK = comm.Get_rank()   # ID de este proceso
SIZE = comm.Get_size()   # Total de procesos

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
    """
    Carga el CSV más reciente cuyo nombre empieza con `prefix`
    desde el directorio datos_crudos/.
    """
    pattern = os.path.join(config.RAW_DATA_DIR, f"{prefix}_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        logger.warning(f"No se encontraron archivos para prefijo '{prefix}'")
        return pd.DataFrame()
    latest = files[-1]
    logger.info(f"Cargando {latest}")
    return pd.read_csv(latest)


def _load_latest_json(prefix: str) -> dict:
    """Carga el JSON más reciente para el prefijo dado."""
    pattern = os.path.join(config.RAW_DATA_DIR, f"{prefix}_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return {}
    with open(files[-1], encoding="utf-8") as f:
        return json.load(f)

#ANÁLISIS DE SENSORES THINGSPEAK

def analizar_thingspeak(df: pd.DataFrame) -> dict:
    """
    Análisis estadístico del trozo de datos de sensores asignado a este proceso.

    Calcula: media, máximo, mínimo, desviación estándar y detección de outliers
    (método IQR) para cada variable ambiental del canal ThingSpeak.
    """
    if df.empty:
        return {"rank": RANK, "error": "DataFrame vacío"}

    resultado = {"rank": RANK, "n_registros": len(df)}

    variables = [
        "temperatura", "humedad", "presion",
        "lluvia", "nivel_voltaje", "nivel_luz",
        "velocidad_viento", "direccion_viento",
    ]

    for var in variables:

        if var not in df.columns:
            continue

        serie = pd.to_numeric(df[var], errors="coerce").dropna()
        if serie.empty:
            continue

        if var == "temperatura":
            serie = (serie - 32) * 5/9  # convertir °F → °C
        # Estadísticas descriptivas básicas
        resultado[f"{var}_prom"] = float(serie.mean())
        resultado[f"{var}_max"]  = float(serie.max())
        resultado[f"{var}_min"]  = float(serie.min())
        resultado[f"{var}_std"]  = float(serie.std())

        # Detección de outliers por IQR
        # Un valor es anómalo si cae fuera de [Q1 - 1.5·IQR, Q3 + 1.5·IQR]
        q1, q3 = serie.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = serie[(serie < q1 - 1.5 * iqr) | (serie > q3 + 1.5 * iqr)]
        resultado[f"{var}_outliers"] = int(len(outliers))


    #Tendencia de temperatura (regresión lineal simple)
    # Si el trozo tiene timestamps, calculamos si la temperatura subió o bajó
    if "created_at" in df.columns and "temperatura" in df.columns:
        df_t = df[["created_at", "temperatura"]].copy()
        df_t["created_at"] = pd.to_datetime(df_t["created_at"], errors="coerce")
        df_t = df_t.dropna()
        if len(df_t) >= 2:
            # Convertir timestamps a segundos desde el inicio del trozo
            t0 = df_t["created_at"].min()
            df_t["t_seg"] = (df_t["created_at"] - t0).dt.total_seconds()
            temp = pd.to_numeric(df_t["temperatura"], errors="coerce").dropna()
            if len(temp) >= 2:
                coef = float(np.polyfit(df_t["t_seg"][:len(temp)], temp, 1)[0])
                resultado["temperatura_tendencia"] = (
                    "subiendo" if coef > 0.001 else
                    "bajando"  if coef < -0.001 else
                    "estable"
                )

    return resultado

#ANÁLISIS DE CONDICIONES METEOROLÓGICAS (OPENWEATHER)

def analizar_openweather(clima: dict) -> dict:
    """
    Evalúa las condiciones meteorológicas y genera indicadores de impacto.

    Determina si las condiciones actuales (visibilidad, viento, nubosidad)
    representan un riesgo ambiental o de visibilidad significativo.
    """
    if not clima:
        return {"error": "Sin datos de OpenWeather"}

    resultado = {
        "ciudad":            clima.get("ciudad"),
        "descripcion":       clima.get("descripcion"),
        "temperatura_c":     clima.get("temperatura"),
        "humedad_pct":       clima.get("humedad"),
        "presion_hpa":       clima.get("presion"),
        "visibilidad_m":     clima.get("visibilidad"),
        "viento_kmh":        clima.get("velocidad_viento"),
        "nubosidad_pct":     clima.get("nubosidad"),
    }

    # Índice de adversidad ambiental (0–100)
    # Combina visibilidad, viento y nubosidad en un solo número
    visib  = float(clima.get("visibilidad", 10000) or 10000)
    viento = float(clima.get("velocidad_viento", 0) or 0)
    nubos  = float(clima.get("nubosidad", 0) or 0)

    adversidad = min(100, (
        (1 - min(visib, 10000) / 10000) * 40 +  # baja visibilidad → peso 40%
        min(viento / 30, 1) * 35 +               # viento fuerte    → peso 35%
        (nubos / 100) * 25                        # nubosidad        → peso 25%
    ))

    resultado["adversidad_ambiental_idx"] = round(adversidad, 2)
    resultado["nivel_adversidad"] = (
        "ALTO"  if adversidad > 60 else
        "MEDIO" if adversidad > 30 else
        "BAJO"
    )

    return resultado

# CORRELACIÓN ENTRE FUENTES (corre solo en el maestro tras el gather)


def correlacion_sensores_clima(
    resultados_thingspeak: list,
    resultado_clima: dict,
) -> dict:
    """
    Cruza los datos de las dos fuentes para responder las preguntas de análisis:

    • ¿Qué condiciones ambientales coinciden con mayor actividad de sensores?
    • ¿El clima actual eleva los niveles de contaminación?
    """
    correlacion = {}

    #Temperatura: ThingSpeak vs OpenWeather
    temps_ts = [r.get("temperatura_prom") for r in resultados_thingspeak
                if r.get("temperatura_prom") is not None]
    if temps_ts and resultado_clima.get("temperatura_c") is not None:
        prom_ts  = float(np.mean(temps_ts))
        prom_ow  = float(resultado_clima["temperatura_c"])
        diferencia = abs(prom_ts - prom_ow)
        correlacion["temperatura_thingspeak_prom"] = round(prom_ts, 2)
        correlacion["temperatura_openweather"]     = prom_ow
        correlacion["diferencia_temperatura_c"]    = round(diferencia, 2)
        correlacion["concordancia_temperatura"] = (
            "BUENA"   if diferencia < 2 else
            "MODERADA" if diferencia < 5 else
            "BAJA"
        )

    #PM2.5: ThingSpeak vs umbral OMS
    pm25_ts = [r.get("pm25_prom") for r in resultados_thingspeak
               if r.get("pm25_prom") is not None]

    #Humedad: ThingSpeak vs OpenWeather
    hums_ts = [r.get("humedad_prom") for r in resultados_thingspeak
               if r.get("humedad_prom") is not None]
    if hums_ts and resultado_clima.get("humedad_pct") is not None:
        correlacion["humedad_thingspeak_prom"] = round(float(np.mean(hums_ts)), 2)
        correlacion["humedad_openweather"]     = resultado_clima["humedad_pct"]

    #Impacto de adversidad climática en contaminación
    adversidad = resultado_clima.get("adversidad_ambiental_idx", 0)
    pm25_global = round(float(np.mean(pm25_ts)), 2) if pm25_ts else 0
    correlacion["interpretacion_ambiental"] = (
        "Alta adversidad climática + contaminación elevada: condiciones críticas"
        if adversidad > 60 and pm25_global > 50
        else "Alta adversidad climática con contaminación moderada"
        if adversidad > 60
        else "Condiciones ambientales dentro de rangos normales"
    )

    return correlacion

# PIPELINE PRINCIPAL MPI

def run_mpi_pipeline():
    """
    Orquesta el procesamiento distribuido completo.

    Proceso 0: maestro → scatter, gather, agregación, guardado.
    Procesos 1..N: trabajadores → reciben datos, analizan, devuelven.
    """

    # FASE 1: El maestro carga los datos y los prepara para distribuir
    if RANK == 0:
        logger.info("═" * 60)
        logger.info(f"MPI Pipeline iniciado con {SIZE} proceso(s)")
        logger.info("═" * 60)

        df_thingspeak = _load_latest_csv("thingspeak")
        clima          = _load_latest_json("weather")

        logger.info(f"ThingSpeak cargado  : {len(df_thingspeak)} registros")
        logger.info(f"OpenWeather cargado : {len(clima)} campos")

        # Dividir ThingSpeak en SIZE trozos (uno por proceso).
        # Usamos iloc con índices numéricos para mantener el tipo DataFrame,
        # y luego convertimos a lista de dicts para que MPI serialice con
        # pickle conservando los nombres de columna.
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
        clima     = None

    # FASE 2: Distribución de datos

    # scatter: cada proceso recibe su lista de dicts.
    # pd.DataFrame(lista_de_dicts) reconstruye las columnas con sus nombres correctos.
    registros_recibidos = comm.scatter(chunks_ts, root=0)
    mi_chunk_ts = pd.DataFrame(registros_recibidos)

    # bcast: todos reciben el mismo clima (es un dict pequeño, no hay que dividirlo)
    clima = comm.bcast(clima, root=0)

    comm.Barrier()
    logger.info(f"Datos recibidos: {len(mi_chunk_ts)} registros ThingSpeak")

    # FASE 3: Análisis local (corre en paralelo en todos los procesos)

    logger.info("Iniciando análisis local...")
    resultado_ts = analizar_thingspeak(mi_chunk_ts)
    logger.info("Análisis local completado.")

    # FASE 4: Gather — recolección de resultados en el maestro

    todos_ts = comm.gather(resultado_ts, root=0)

    # FASE 5: Agregación final (solo el maestro)

    if RANK == 0:
        logger.info("Agregando resultados de todos los procesos...")

        todos_ts_ok = [r for r in todos_ts if "error" not in r]

        # Análisis del clima (no necesita agregación, es un dict único)
        resultado_clima = analizar_openweather(clima)

        # Correlación cruzada entre las dos fuentes
        correlacion = correlacion_sensores_clima(todos_ts_ok, resultado_clima)

        # Función auxiliar para promediar una métrica entre todos los procesos
        def prom_global(key):
            vals = [r[key] for r in todos_ts_ok if key in r]
            return round(float(np.mean(vals)), 2) if vals else None

        # Conteo total de outliers detectados entre todos los procesos
        variables = [
            "temperatura", "humedad", "presion",
            "lluvia", "nivel_voltaje", "nivel_luz", "velocidad_viento",
        ]
        outliers_global = {
            var: sum(r.get(f"{var}_outliers", 0) for r in todos_ts_ok)
            for var in variables
        }

        # Alertas de PM2.5 (tomamos la más severa entre todos los procesos)
        alertas_pm25 = [r.get("alerta_pm25", "") for r in todos_ts_ok]
        alerta_final = (
            next((a for a in alertas_pm25 if "PELIGROSO" in a), None)
            or next((a for a in alertas_pm25 if "MODERADO"  in a), None)
            or "NORMAL"
        )

        # Armar reporte final
        reporte = {
            "metadata": {
                "timestamp":      datetime.utcnow().isoformat(),
                "procesos_mpi":   SIZE,
                "total_registros_thingspeak": sum(
                    r.get("n_registros", 0) for r in todos_ts_ok
                ),
            },
            "thingspeak": {
                "temperatura_prom_global":      prom_global("temperatura_prom"),
                "humedad_prom_global":          prom_global("humedad_prom"),
                "presion_prom_global":          prom_global("presion_prom"),
                "velocidad_viento_prom_global": prom_global("velocidad_viento_prom"),
                "direccion_viento_prom_global": prom_global("direccion_viento_prom"),
                "lluvia_prom_global":           prom_global("lluvia_prom"),
                "nivel_luz_prom_global":        prom_global("nivel_luz_prom"),
                "alerta_pm25":                 alerta_final,
                "outliers_detectados":         outliers_global,
                "tendencias_temperatura":      [
                    r.get("temperatura_tendencia") for r in todos_ts_ok
                    if r.get("temperatura_tendencia")
                ],
                "resultados_por_proceso":      todos_ts_ok,
            },
            "openweather":       resultado_clima,
            "correlacion_fuentes": correlacion,
        }

        # Guardar reporte
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        reporte_path = os.path.join(config.PROCESSED_DIR, f"reporte_mpi_{ts}.json")
        with open(reporte_path, "w", encoding="utf-8") as f:
            json.dump(reporte, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Reporte guardado en: {reporte_path}")

        # Imprimir resumen en consola
        print("\n" + "═" * 60)
        print("  RESUMEN DEL ANÁLISIS MPI — DATOS AMBIENTALES")
        print("═" * 60)
        print(f"  Procesos MPI                : {SIZE}")
        print(f"  Registros ThingSpeak        : {reporte['metadata']['total_registros_thingspeak']}")
        print(f"  Temperatura prom. (TS)      : {reporte['thingspeak']['temperatura_prom_global']} C")
        print(f"  Temperatura (OpenWeather)   : {resultado_clima.get('temperatura_c')} C")
        print(f"  Concordancia temp.          : {correlacion.get('concordancia_temperatura', 'N/A')}")
        print(f"  Humedad prom. (TS)          : {reporte['thingspeak']['humedad_prom_global']} %")
        print(f"  Velocidad viento prom. (TS) : {reporte['thingspeak']['velocidad_viento_prom_global']} mph")
        print(f"  Lluvia prom. (TS)           : {reporte['thingspeak']['lluvia_prom_global']} in/min")
        print(f"  Nivel luz prom. (TS)        : {reporte['thingspeak']['nivel_luz_prom_global']}")
        print(f"  Adversidad ambiental        : {resultado_clima.get('adversidad_ambiental_idx')}/100")
        print(f"  Nivel adversidad            : {resultado_clima.get('nivel_adversidad')}")
        print(f"  Interpretacion              : {correlacion.get('interpretacion_ambiental')}")
        print(f"  Outliers temperatura        : {outliers_global.get('temperatura', 0)}")
        print(f"  Outliers velocidad viento   : {outliers_global.get('velocidad_viento', 0)}")
        print("═" * 60)
        print(f"  Reporte → {reporte_path}")
        print("═" * 60 + "\n")

        return reporte

    return None

# PUNTO DE ENTRADA

if __name__ == "__main__":
    if RANK == 0:
        raw_files = glob.glob(os.path.join(config.RAW_DATA_DIR, "*.csv"))
        if not raw_files:
            logger.error(
                "No se encontraron datos crudos. "
                "Ejecuta primero: python data_fetcher.py"
            )
            comm.Abort(1)
            sys.exit(1)

    reporte = run_mpi_pipeline()
    MPI.Finalize()