import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

import aiohttp
import pandas as pd

import config


logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("DataFetcher")


class APIClient:
    """Cliente HTTP asincrono con timeout, limite de conexiones y reintentos."""

    def __init__(
        self,
        timeout: int = config.REQUEST_TIMEOUT,
        max_retries: int = config.MAX_RETRIES,
    ):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
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

    async def get(self, url: str, params: dict | None = None, label: str = "") -> Any:
        """Realiza una peticion GET asincrona con reintentos exponenciales."""
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info("[%s] GET %s (intento %s)", label, url, attempt)
                async with self._session.get(url, params=params) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    logger.info("[%s] Respuesta recibida (status %s)", label, resp.status)
                    return data

            except aiohttp.ClientResponseError as exc:
                logger.warning("[%s] Error HTTP %s: %s", label, exc.status, exc.message)
            except aiohttp.ClientConnectionError as exc:
                logger.warning("[%s] Error de conexion: %s", label, exc)
            except asyncio.TimeoutError:
                logger.warning("[%s] Timeout en intento %s", label, attempt)
            except Exception as exc:
                logger.error("[%s] Error inesperado: %s", label, exc)

            if attempt < self.max_retries:
                wait = config.RETRY_DELAY * (2 ** (attempt - 1))
                logger.info("[%s] Reintentando en %ss...", label, wait)
                await asyncio.sleep(wait)

        logger.error("[%s] Todos los intentos fallaron.", label)
        return None


def _thingspeak_dataframe(raw: dict, label: str) -> pd.DataFrame:
    feeds = raw.get("feeds") or []
    channel = raw.get("channel") or {}

    if not feeds:
        fields = {
            key: value
            for key, value in channel.items()
            if key.startswith("field") and value
        }
        logger.warning(
            "[%s] La API respondio, pero no devolvio lecturas. feeds=0, campos=%s",
            label,
            fields or "sin campos publicados",
        )
        return pd.DataFrame()

    df = pd.DataFrame(feeds)
    if "created_at" not in df.columns:
        logger.warning(
            "[%s] La respuesta no contiene columna created_at. Columnas: %s",
            label,
            list(df.columns),
        )
        return pd.DataFrame()

    df.rename(columns=config.THINGSPEAK_FIELD_NAMES, inplace=True)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    for col in df.columns:
        if col != "created_at":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.sort_values("created_at", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


async def _fetch_thingspeak_channel(
    client: APIClient,
    channel_id: int,
    read_key: str = "",
    label: str = "ThingSpeak",
) -> pd.DataFrame:
    url = f"{config.THINGSPEAK_BASE_URL}/channels/{channel_id}/feeds.json"
    params = {"results": config.THINGSPEAK_NUM_RESULTS}
    if read_key:
        params["api_key"] = read_key

    raw = await client.get(url, params=params, label=label)
    if not raw or "feeds" not in raw:
        logger.warning("[%s] Respuesta vacia o sin arreglo feeds.", label)
        return pd.DataFrame()

    return _thingspeak_dataframe(raw, label)


async def fetch_thingspeak(client: APIClient) -> Optional[pd.DataFrame]:
    """
    Obtiene las ultimas lecturas de ThingSpeak.

    Primero intenta usar el canal/key de keys.py. Si ese canal no devuelve
    lecturas, usa un canal publico de respaldo para que el pipeline siga siendo
    ejecutable y deje un diagnostico claro.
    """
    df = await _fetch_thingspeak_channel(
        client,
        config.THINGSPEAK_CHANNEL_ID,
        config.THINGSPEAK_READ_KEY,
        label=f"ThingSpeak:{config.THINGSPEAK_CHANNEL_ID}",
    )

    if not df.empty:
        logger.info("[ThingSpeak] %s registros cargados.", len(df))
        return df

    fallback_id = config.THINGSPEAK_FALLBACK_CHANNEL_ID
    if (
        config.THINGSPEAK_ALLOW_PUBLIC_FALLBACK
        and fallback_id
        and fallback_id != config.THINGSPEAK_CHANNEL_ID
    ):
        logger.warning(
            "[ThingSpeak] El canal configurado no tiene lecturas disponibles. "
            "Se usara el canal publico de respaldo %s para completar el pipeline. "
            "Para usar tu canal, revisa que tenga datos y que thingspeak_api sea "
            "una Read API Key valida.",
            fallback_id,
        )
        df = await _fetch_thingspeak_channel(
            client,
            fallback_id,
            "",
            label=f"ThingSpeak:fallback:{fallback_id}",
        )
        if not df.empty:
            logger.info("[ThingSpeak] %s registros cargados desde fallback.", len(df))
            return df

    raise RuntimeError(
        "ThingSpeak no devolvio lecturas. Revisa thingspeak_channelID, "
        "thingspeak_api (debe ser Read API Key) y que el canal tenga entradas."
    )


async def fetch_openweather(client: APIClient) -> Optional[dict]:
    """Obtiene condiciones meteorologicas actuales desde OpenWeather."""
    if not config.OPENWEATHER_API_KEY:
        logger.warning(
            "[OpenWeather] Sin API key. Define OPENWEATHER_API_KEY o crea keys.py; "
            "se omitira esta fuente y el pipeline continuara con ThingSpeak."
        )
        return None

    url = f"{config.OPENWEATHER_BASE_URL}/weather"
    params = {
        "q": config.OPENWEATHER_CITY,
        "appid": config.OPENWEATHER_API_KEY,
        "units": config.OPENWEATHER_UNITS,
        "lang": "es",
    }
    raw = await client.get(url, params=params, label="OpenWeather")
    if not raw:
        return None

    weather = {
        "ciudad": raw.get("name"),
        "temperatura": raw.get("main", {}).get("temp"),
        "sensacion_termica": raw.get("main", {}).get("feels_like"),
        "humedad": raw.get("main", {}).get("humidity"),
        "presion": raw.get("main", {}).get("pressure"),
        "visibilidad": raw.get("visibility"),
        "descripcion": raw.get("weather", [{}])[0].get("description"),
        "velocidad_viento": raw.get("wind", {}).get("speed"),
        "direccion_viento": raw.get("wind", {}).get("deg"),
        "nubosidad": raw.get("clouds", {}).get("all"),
        "timestamp": datetime.fromtimestamp(raw.get("dt", 0), timezone.utc),
    }
    logger.info(
        "[OpenWeather] Clima obtenido: %s, %s C",
        weather["descripcion"],
        weather["temperatura"],
    )
    return weather


async def fetch_all_sources() -> dict[str, Any]:
    """Ejecuta todas las peticiones de forma concurrente y guarda datos crudos."""
    start = time.perf_counter()
    logger.info("=" * 60)
    logger.info("Iniciando fetching concurrente de todas las fuentes...")

    async with APIClient() as client:
        df_thingspeak, weather = await asyncio.gather(
            fetch_thingspeak(client),
            fetch_openweather(client),
        )

    elapsed = time.perf_counter() - start
    logger.info("Fetching completado en %.2fs", elapsed)
    logger.info("=" * 60)

    results = {
        "thingspeak": df_thingspeak,
        "weather": weather,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    _save_raw(results)
    return results


def _save_raw(data: dict) -> None:
    """Guarda cada fuente en datos_crudos/ para que MPI la lea desde disco."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    for key, value in data.items():
        if key == "timestamp":
            continue

        path_base = os.path.join(config.RAW_DATA_DIR, f"{key}_{ts}")

        if isinstance(value, pd.DataFrame) and not value.empty:
            csv_path = path_base + ".csv"
            value.to_csv(csv_path, index=False)
            logger.info("[Save] %s -> %s", key, csv_path)

        elif isinstance(value, dict):
            json_path = path_base + ".json"
            with open(json_path, "w", encoding="utf-8") as file:
                clean = {
                    k: v.isoformat() if isinstance(v, datetime) else v
                    for k, v in value.items()
                }
                json.dump(clean, file, ensure_ascii=False, indent=2)
            logger.info("[Save] %s -> %s", key, json_path)

        else:
            logger.warning("[Save] %s: sin datos o formato no reconocido.", key)


if __name__ == "__main__":
    results = asyncio.run(fetch_all_sources())

    print("\n--- Resumen de datos obtenidos ---")
    for key, val in results.items():
        if isinstance(val, pd.DataFrame):
            print(f"  {key:15s}: DataFrame con {len(val)} filas y {len(val.columns)} columnas")
        elif isinstance(val, dict):
            print(f"  {key:15s}: dict con {len(val)} campos")
        else:
            print(f"  {key:15s}: {val}")
