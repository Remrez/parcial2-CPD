import os

try:
    import keys
except ImportError:
    keys = None


def _key_value(*names: str, default: str = ""):
    """Busca una credencial en variables de entorno o en keys.py."""
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return value

    if keys is None:
        return default

    for name in names:
        value = getattr(keys, name, None)
        if value not in (None, ""):
            return value

    return default


def _key_int(*names: str, default: int) -> int:
    value = _key_value(*names, default=str(default))
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# Credenciales.
#
# Compatibilidad con el keys.py original del proyecto:
#   openweather_api
#   thingspeak_api
#   thingspeak_channelID
OPENWEATHER_API_KEY = _key_value(
    "OPENWEATHER_API_KEY",
    "openweather_api",
)

THINGSPEAK_CHANNEL_ID = _key_int(
    "THINGSPEAK_CHANNEL_ID",
    "thingspeak_channelID",
    "thingspeak_channel_id",
    default=12397,
)

THINGSPEAK_READ_KEY = _key_value(
    "THINGSPEAK_READ_KEY",
    "THINGSPEAK_API_KEY",
    "thingspeak_api",
)

THINGSPEAK_NUM_RESULTS = int(os.getenv("THINGSPEAK_NUM_RESULTS", "100"))
THINGSPEAK_FALLBACK_CHANNEL_ID = _key_int(
    "THINGSPEAK_FALLBACK_CHANNEL_ID",
    default=12397,
)
THINGSPEAK_ALLOW_PUBLIC_FALLBACK = (
    os.getenv("THINGSPEAK_ALLOW_PUBLIC_FALLBACK", "1").strip().lower()
    not in {"0", "false", "no"}
)

THINGSPEAK_FIELD_NAMES = {
    "field1": "direccion_viento",   # grados
    "field2": "velocidad_viento",   # mph
    "field3": "humedad",            # %
    "field4": "temperatura",        # F
    "field5": "lluvia",             # inches/min
    "field6": "presion",            # inHg
    "field7": "nivel_voltaje",      # V
    "field8": "nivel_luz",          # intensidad
}

THINGSPEAK_BASE_URL = "https://api.thingspeak.com"
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"

# Parametros de consulta OpenWeather.
OPENWEATHER_CITY = os.getenv("OPENWEATHER_CITY", "Mexico City,MX")
OPENWEATHER_UNITS = os.getenv("OPENWEATHER_UNITS", "metric")

# Concurrencia Asyncio.
REQUEST_TIMEOUT = 15
MAX_CONNECTIONS_PER_HOST = 5
RETRY_DELAY = 2
MAX_RETRIES = 3

# Rutas de archivos de salida.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "datos_crudos")
PROCESSED_DIR = os.path.join(BASE_DIR, "datos_procesados")
REPORTS_DIR = os.path.join(BASE_DIR, "reportes")

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Logging.
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(BASE_DIR, "pipeline.log")
