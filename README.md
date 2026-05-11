# Parcial 2 - Computo Paralelo y Distribuido

Pipeline de analisis ambiental que integra las tres tecnologias pedidas en la
actividad:

- **Asyncio**: consulta concurrente de APIs.
- **MPI4Py**: distribucion del procesamiento por procesos.
- **CUDA/CuPy**: analisis matricial acelerado por GPU NVIDIA.

El resultado final es un archivo HTML listo para presentar.

## Requisitos

1. Python 3.10 o superior.
2. Microsoft MPI en Windows:
   <https://learn.microsoft.com/es-es/message-passing-interface/microsoft-mpi>
3. Dependencias base:

```powershell
pip install -r requirements.txt
```

4. Para ejecutar la fase CUDA real en GPU NVIDIA:

```powershell
pip install -r requirements-gpu.txt
```

Si no hay GPU/CuPy disponible, el programa usa un fallback CPU y lo deja
registrado en `reporte_cuda_*.json`. Para defender la parte CUDA conviene
correrlo en una maquina con driver NVIDIA y CuPy instalado.

Resultado esperado:

- En una computadora sin NVIDIA: `Backend : cpu_fallback`.
- En una computadora con NVIDIA/CUDA funcionando: `Backend : cuda`.

Si en Windows aparece `Failure finding "nvrtc*.dll"`, significa que CuPy no
encuentra NVRTC, la libreria de compilacion runtime de CUDA. Soluciones:

```powershell
pip install -U "cupy-cuda12x[ctk]"
```

o agrega el binario de CUDA al entorno antes de ejecutar:

```powershell
$env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$env:PATH="$env:CUDA_PATH\bin;$env:PATH"
python main.py 4
```

La carpeta exacta puede variar (`v12.4`, `v12.5`, `v12.6`, etc.).

## API key de OpenWeather

El proyecto puede correr solo con ThingSpeak. Para incluir OpenWeather, usa una
de estas opciones:

Opcion A: variable de entorno

```powershell
$env:OPENWEATHER_API_KEY="TU_API_KEY"
```

Opcion B: archivo local `keys.py` ignorado por git

```python
openweather_api = "TU_API_KEY"
thingspeak_api = "TU_THINGSPEAK_READ_API_KEY"
thingspeak_channelID = 123456
```

El codigo tambien acepta variables de entorno (`OPENWEATHER_API_KEY`,
`THINGSPEAK_READ_KEY`, `THINGSPEAK_CHANNEL_ID`), pero para este proyecto el
formato anterior de `keys.py` sigue funcionando.

Nota: `thingspeak_api` debe ser una **Read API Key**. Si el canal no tiene
lecturas o la key no permite leer datos, el programa lo avisara y usara el canal
publico `12397` como respaldo para que el pipeline completo pueda ejecutarse.

## Ejecucion completa

```powershell
python main.py 4
```

El numero `4` indica cuantos procesos MPI se lanzan. Puedes cambiarlo segun los
nucleos disponibles.

## Salidas generadas

- `datos_crudos/thingspeak_*.csv`: datos obtenidos por Asyncio.
- `datos_crudos/weather_*.json`: clima actual si hay API key.
- `datos_procesados/reporte_mpi_*.json`: resultados agregados por MPI.
- `datos_procesados/reporte_cuda_*.json`: matriz, correlaciones y criticidad
  calculadas con CUDA o fallback CPU.
- `reportes/reporte_final.html`: reporte final para abrir en el navegador.

Para la entrega en PDF, abre `reportes/reporte_final.html` en el navegador y
usa **Imprimir > Guardar como PDF**.

## Ejecucion por fases

```powershell
python data_fetcher.py
mpiexec -n 4 python mpi_processor.py
python cuda_processor.py
python report_generator.py
```

## Que hace la fase CUDA

La fase 3 toma las metricas que produjo MPI por cada rank y forma una matriz
con temperatura, humedad, presion, viento, lluvia y nivel de luz. Con CuPy:

- mueve la matriz a memoria GPU,
- calcula medias, desviaciones y z-scores,
- calcula una matriz de correlacion,
- ejecuta un `ElementwiseKernel` CUDA para asignar un score de criticidad
  relativa por bloque MPI.

Esto conecta naturalmente el trabajo de tu companera con una parte propia de
CUDA y deja evidencia en JSON y HTML.
