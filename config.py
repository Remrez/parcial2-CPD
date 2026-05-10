import keys

OPENWEATHER_API_KEY = keys.openweather_api  

THINGSPEAK_CHANNEL_ID  = 12397               
THINGSPEAK_READ_KEY    = ""                     
THINGSPEAK_NUM_RESULTS = 100                    

THINGSPEAK_FIELD_NAMES = {
    "field1": "direccion_viento",   # grados (0 = Norte)
    "field2": "velocidad_viento",   # mph
    "field3": "humedad",            # %
    "field4": "temperatura",        # °F 
    "field5": "lluvia",             # inches/min
    "field6": "presion",            # inHg (no hPa)
    "field7": "nivel_voltaje",      # V
    "field8": "nivel_luz",          # intensidad
}

THINGSPEAK_BASE_URL  = "https://api.thingspeak.com"
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"

# ── Parámetros de consulta ─────────────────────────────────────────────────────
# OpenWeather: ciudad o coordenadas de referencia
OPENWEATHER_CITY  = "Mexico City,MX"
OPENWEATHER_UNITS = "metric"   # metric = Celsius, imperial = Fahrenheit

# ── Concurrencia (Asyncio) ─────────────────────────────────────────────────────
# Tiempo máximo de espera por solicitud HTTP (segundos)
REQUEST_TIMEOUT = 15

# Límite de conexiones simultáneas al mismo host (controla la presión sobre APIs)
MAX_CONNECTIONS_PER_HOST = 5

# Pausa entre reintentos en caso de error (segundos)
RETRY_DELAY = 2

# Número de reintentos por solicitud fallida
MAX_RETRIES = 3

# ── Rutas de archivos de salida ────────────────────────────────────────────────
import os
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR   = os.path.join(BASE_DIR, "datos_crudos")
PROCESSED_DIR  = os.path.join(BASE_DIR, "datos_procesados")

# Crear directorios si no existen
os.makedirs(RAW_DATA_DIR,  exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"   # DEBUG, INFO, WARNING, ERROR
LOG_FILE  = os.path.join(BASE_DIR, "pipeline.log")