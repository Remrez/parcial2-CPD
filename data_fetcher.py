import asyncio
import aiohttp
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Optional

import pandas as pd

import config

# ── Configuración del logger ───────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("DataFetcher")

class APIClient:
    """
    Cliente HTTP asíncrono reutilizable.

    Parámetros
    ----------
    timeout : int
        Segundos máximos de espera por solicitud.
    max_retries : int
        Cuántas veces reintentar si falla la solicitud.
    """

    def __init__(
        self,
        timeout: int = config.REQUEST_TIMEOUT,
        max_retries: int = config.MAX_RETRIES,
    ):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None

    # Usamos "async with APIClient() as client:" para garantizar que la sesión
    # HTTP siempre se cierre aunque ocurra una excepción.

    async def __aenter__(self):
        # TCPConnector limita conexiones simultáneas al mismo host
        connector = aiohttp.TCPConnector(
            limit_per_host=config.MAX_CONNECTIONS_PER_HOST
        )
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=self.timeout,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()

    # ── Método base de petición con reintentos ───────────────────────────────

    async def get(self, url: str, params: dict = None, label: str = "") -> Any:
        """
        Realiza una petición GET asíncrona con reintentos exponenciales.

        Retorna el JSON parseado o None si todos los reintentos fallan.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"[{label}] GET {url} (intento {attempt})")
                async with self._session.get(url, params=params) as resp:
                    resp.raise_for_status()           # lanza excepción si status ≥ 400
                    data = await resp.json()
                    logger.info(f"[{label}] ✓ Respuesta recibida (status {resp.status})")
                    return data

            except aiohttp.ClientResponseError as e:
                logger.warning(f"[{label}] Error HTTP {e.status}: {e.message}")
            except aiohttp.ClientConnectionError as e:
                logger.warning(f"[{label}] Error de conexión: {e}")
            except asyncio.TimeoutError:
                logger.warning(f"[{label}] Timeout en intento {attempt}")
            except Exception as e:
                logger.error(f"[{label}] Error inesperado: {e}")

            if attempt < self.max_retries:
                # Backoff exponencial: 2s, 4s, 8s, ...
                wait = config.RETRY_DELAY * (2 ** (attempt - 1))
                logger.info(f"[{label}] Reintentando en {wait}s...")
                await asyncio.sleep(wait)

        logger.error(f"[{label}] Todos los intentos fallaron.")
        return None


# Cada función es una corrutina (async def) que usa el cliente compartido.
# Reciben el cliente como parámetro para reusar la misma sesión HTTP.

#THINGSPEAK

async def fetch_thingspeak(client: APIClient) -> Optional[pd.DataFrame]:
    """
    Obtiene las últimas N lecturas de un canal ThingSpeak.

    ThingSpeak estructura su respuesta como:
    {
      "channel": { ...metadata... },
      "feeds": [ {"created_at": "...", "field1": "...", ...}, ... ]
    }

    Es la fuente de datos principal del proyecto: provee temperatura,
    humedad, presión, PM2.5, PM10, nivel de luz y datos de viento.
    """
    url = (
        f"{config.THINGSPEAK_BASE_URL}"
        f"/channels/{config.THINGSPEAK_CHANNEL_ID}/feeds.json"
    )
    params = {"results": config.THINGSPEAK_NUM_RESULTS}
    if config.THINGSPEAK_READ_KEY:
        params["api_key"] = config.THINGSPEAK_READ_KEY

    raw = await client.get(url, params=params, label="ThingSpeak")
    if not raw or "feeds" not in raw:
        return None

    feeds = raw["feeds"]
    df = pd.DataFrame(feeds)

    # Renombrar columnas genéricas (field1, field2…) a nombres descriptivos
    df.rename(columns=config.THINGSPEAK_FIELD_NAMES, inplace=True)

    # Convertir timestamp a datetime y campos numéricos
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    for col in df.columns:
        if col != "created_at":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.sort_values("created_at", inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(f"[ThingSpeak] {len(df)} registros cargados.")
    return df


#OPENWEATHER

async def fetch_openweather(client: APIClient) -> Optional[dict]:
    """
    Obtiene condiciones meteorológicas actuales.

    Retorna un dict con temperatura, humedad, presión, descripción del clima,
    velocidad del viento, visibilidad, etc.

    Estos datos se cruzan con los sensores de ThingSpeak para validar lecturas
    y con OpenAQ para enriquecer el análisis ambiental.
    """
    url = f"{config.OPENWEATHER_BASE_URL}/weather"
    params = {
        "q":     config.OPENWEATHER_CITY,
        "appid": config.OPENWEATHER_API_KEY,
        "units": config.OPENWEATHER_UNITS,
        "lang":  "es",
    }
    raw = await client.get(url, params=params, label="OpenWeather")
    if not raw:
        return None

    # Aplanamos la respuesta anidada en un dict plano fácil de usar
    weather = {
        "ciudad":             raw.get("name"),
        "temperatura":        raw.get("main", {}).get("temp"),
        "sensacion_termica":  raw.get("main", {}).get("feels_like"),
        "humedad":            raw.get("main", {}).get("humidity"),
        "presion":            raw.get("main", {}).get("pressure"),
        "visibilidad":        raw.get("visibility"),          # metros
        "descripcion":        raw.get("weather", [{}])[0].get("description"),
        "velocidad_viento":   raw.get("wind", {}).get("speed"),
        "direccion_viento":   raw.get("wind", {}).get("deg"),
        "nubosidad":          raw.get("clouds", {}).get("all"),  # %
        "timestamp":          datetime.utcfromtimestamp(raw.get("dt", 0)),
    }
    logger.info(
        f"[OpenWeather] Clima obtenido: "
        f"{weather['descripcion']}, {weather['temperatura']}°C"
    )
    return weather

# Esta es la función que el resto del pipeline llama.
# Lanza TODAS las fuentes al mismo tiempo con asyncio.gather.

async def fetch_all_sources() -> dict[str, Any]:
    """
    Ejecuta todas las peticiones de forma concurrente.

    Retorna un diccionario con los DataFrames/dicts de cada fuente:
    {
        "thingspeak":  DataFrame,   ← datos de sensores (fuente principal)
        "weather":     dict,        ← condiciones meteorológicas actuales
        "timestamp":   str,
    }
    """
    start = time.perf_counter()
    logger.info("═" * 60)
    logger.info("Iniciando fetching concurrente de todas las fuentes...")

    async with APIClient() as client:
        # asyncio.gather lanza todas las corrutinas simultáneamente.
        # Cada una puede estar esperando su respuesta HTTP al mismo tiempo,
        # sin que ninguna bloquee a las demás.
        (
            df_thingspeak,
            weather,
        ) = await asyncio.gather(
            fetch_thingspeak(client),
            fetch_openweather(client),
        )

    elapsed = time.perf_counter() - start
    logger.info(f"Fetching completado en {elapsed:.2f}s")
    logger.info("═" * 60)

    results = {
        "thingspeak":  df_thingspeak,
        "weather":     weather,
        "timestamp":   datetime.utcnow().isoformat(),
    }

    # Guardar datos crudos en CSV/JSON para que MPI los lea desde disco
    _save_raw(results)

    return results

# Los procesos MPI no comparten memoria con el proceso Asyncio, así que
# guardamos los datos en archivos JSON/CSV que MPI leerá después.

def _save_raw(data: dict) -> None:
    """Guarda cada fuente en un archivo CSV/JSON dentro de datos_crudos/."""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    for key, value in data.items():
        if key == "timestamp":
            continue
        path_base = os.path.join(config.RAW_DATA_DIR, f"{key}_{ts}")

        if isinstance(value, pd.DataFrame) and not value.empty:
            csv_path = path_base + ".csv"
            value.to_csv(csv_path, index=False)
            logger.info(f"[Save] {key} → {csv_path}")

        elif isinstance(value, dict):
            json_path = path_base + ".json"
            with open(json_path, "w", encoding="utf-8") as f:
                clean = {
                    k: str(v) if isinstance(v, datetime) else v
                    for k, v in value.items()
                }
                json.dump(clean, f, ensure_ascii=False, indent=2)
            logger.info(f"[Save] {key} → {json_path}")

        else:
            logger.warning(f"[Save] {key}: sin datos o formato no reconocido.")


# PUNTO DE ENTRADA DIRECTO (para pruebas del módulo)

if __name__ == "__main__":
    # asyncio.run() crea el event loop, ejecuta la corrutina y lo cierra al final
    results = asyncio.run(fetch_all_sources())

    print("\n── Resumen de datos obtenidos ──────────────────────────────")
    for key, val in results.items():
        if isinstance(val, pd.DataFrame):
            print(
                f"  {key:15s}: DataFrame con {len(val)} filas "
                f"y {len(val.columns)} columnas"
            )
        elif isinstance(val, dict):
            print(f"  {key:15s}: dict con {len(val)} campos")
        else:
            print(f"  {key:15s}: {val}")