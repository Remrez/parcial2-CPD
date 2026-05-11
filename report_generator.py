import argparse
import glob
import html
import json
import os
from datetime import datetime
from typing import Any

import config


def _latest(prefix: str, required: bool = True) -> str | None:
    files = sorted(glob.glob(os.path.join(config.PROCESSED_DIR, f"{prefix}_*.json")))
    if files:
        return files[-1]
    if required:
        raise FileNotFoundError(
            f"No se encontro ningun archivo {prefix}_*.json en {config.PROCESSED_DIR}"
        )
    return None


def _load_json(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _fmt(value: Any, suffix: str = "", digits: int = 2) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{numeric:.{digits}f}{suffix}"


def _risk_color(level: str) -> str:
    return {
        "ALTO": "#b42318",
        "MEDIO": "#b54708",
        "BAJO": "#087443",
    }.get(level, "#345")


def _bar(value: Any, level: str = "BAJO") -> str:
    try:
        width = max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        width = 0.0
    color = _risk_color(level)
    return (
        '<div class="bar" aria-label="score">'
        f'<span style="width:{width:.1f}%; background:{color};"></span>'
        "</div>"
    )


def _metric_card(label: str, value: Any, note: str = "") -> str:
    return (
        '<article class="metric">'
        f"<span>{_e(label)}</span>"
        f"<strong>{_e(value)}</strong>"
        f"<small>{_e(note)}</small>"
        "</article>"
    )


def _stats_rows(stats: dict) -> str:
    rows = []
    for item in stats.values():
        rows.append(
            "<tr>"
            f"<td>{_e(item.get('label'))}</td>"
            f"<td>{_fmt(item.get('mean'), ' ' + item.get('unit', ''))}</td>"
            f"<td>{_fmt(item.get('std'))}</td>"
            f"<td>{_fmt(item.get('min'), ' ' + item.get('unit', ''))}</td>"
            f"<td>{_fmt(item.get('max'), ' ' + item.get('unit', ''))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _risk_rows(rows: list[dict]) -> str:
    html_rows = []
    for row in rows:
        level = row.get("nivel", "BAJO")
        score = row.get("score", 0)
        html_rows.append(
            "<tr>"
            f"<td>Rank {_e(row.get('rank'))}</td>"
            f"<td>{_bar(score, level)}</td>"
            f"<td><strong style=\"color:{_risk_color(level)}\">{_fmt(score)}</strong></td>"
            f"<td>{_e(level)}</td>"
            "</tr>"
        )
    return "\n".join(html_rows)


def _corr_color(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    alpha = min(0.82, 0.12 + abs(numeric) * 0.6)
    if numeric >= 0:
        return f"rgba(20, 110, 92, {alpha:.2f})"
    return f"rgba(180, 35, 24, {alpha:.2f})"


def _corr_table(correlation: dict) -> str:
    labels = correlation.get("labels") or []
    matrix = correlation.get("matrix") or []
    if not labels or not matrix:
        return "<p>No hay matriz de correlacion disponible.</p>"

    header = "<tr><th></th>" + "".join(f"<th>{_e(label)}</th>" for label in labels) + "</tr>"
    rows = [header]
    for label, values in zip(labels, matrix):
        cells = [f"<th>{_e(label)}</th>"]
        for value in values:
            cells.append(
                f'<td style="background:{_corr_color(value)}">{_fmt(value, digits=2)}</td>'
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return '<table class="corr">' + "\n".join(rows) + "</table>"


def _interpretation(items: list[str]) -> str:
    if not items:
        return "<p>No hay interpretacion generada.</p>"
    return "<ul>" + "".join(f"<li>{_e(item)}</li>" for item in items) + "</ul>"


def generate_html_report(
    mpi_report: str | None = None,
    cuda_report: str | None = None,
    output_path: str | None = None,
) -> str:
    mpi_path = mpi_report or _latest("reporte_mpi")
    cuda_path = cuda_report or _latest("reporte_cuda", required=False)
    output = output_path or os.path.join(config.REPORTS_DIR, "reporte_final.html")

    mpi = _load_json(mpi_path)
    cuda = _load_json(cuda_path)

    mpi_meta = mpi.get("metadata", {})
    ts = mpi.get("thingspeak", {})
    weather = mpi.get("openweather", {})
    corr_sources = mpi.get("correlacion_fuentes", {})
    cuda_meta = cuda.get("metadata", {})

    backend = cuda_meta.get("backend", "pendiente")
    device = cuda_meta.get("device", "N/A")
    fallback = cuda_meta.get("fallback_reason")
    risk_rows = cuda.get("riesgo_relativo_por_bloque", [])
    max_risk = max(risk_rows, key=lambda row: row.get("score", 0)) if risk_rows else None

    cards = [
        _metric_card("Registros ThingSpeak", mpi_meta.get("total_registros_thingspeak", "N/A"), "Asyncio"),
        _metric_card("Procesos MPI", mpi_meta.get("procesos_mpi", "N/A"), "scatter/gather"),
        _metric_card("Backend fase 3", backend, device),
        _metric_card(
            "Mayor criticidad",
            f"{max_risk.get('score')} ({max_risk.get('nivel')})" if max_risk else "N/A",
            f"rank {max_risk.get('rank')}" if max_risk else "",
        ),
    ]

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    source_note = (
        f"MPI: {_e(os.path.basename(mpi_path))}"
        + (f" | CUDA: {_e(os.path.basename(cuda_path))}" if cuda_path else " | CUDA: pendiente")
    )

    fallback_html = (
        f'<p class="notice"><strong>Nota CUDA:</strong> {_e(fallback)}</p>' if fallback else ""
    )

    html_doc = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Reporte CPD - Analisis paralelo ambiental</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #18212f;
      --muted: #667085;
      --line: #d7dde8;
      --surface: #ffffff;
      --soft: #f4f7fb;
      --blue: #2457a6;
      --green: #087443;
      --amber: #b54708;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: var(--soft);
      line-height: 1.5;
    }}
    header {{
      background: linear-gradient(135deg, #123c69 0%, #146e5c 100%);
      color: white;
      padding: 40px min(7vw, 72px) 34px;
    }}
    header p {{ max-width: 860px; margin: 10px 0 0; color: rgba(255,255,255,.86); }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0; font-size: clamp(30px, 5vw, 54px); letter-spacing: 0; }}
    h2 {{ margin: 0 0 14px; font-size: 24px; letter-spacing: 0; }}
    h3 {{ margin: 18px 0 8px; font-size: 18px; letter-spacing: 0; }}
    section {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 22px;
      margin: 18px 0;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-top: -54px;
    }}
    .metric {{
      background: white;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      min-height: 122px;
      box-shadow: 0 12px 30px rgba(16, 24, 40, .08);
    }}
    .metric span, .metric small {{ display: block; color: var(--muted); }}
    .metric strong {{ display: block; margin: 8px 0; font-size: 26px; overflow-wrap: anywhere; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; }}
    th {{ color: #344054; background: #f8fafc; }}
    .corr th, .corr td {{ text-align: center; min-width: 92px; }}
    .corr td {{ color: #111827; font-weight: 700; }}
    .bar {{ height: 10px; background: #edf1f7; border-radius: 999px; overflow: hidden; }}
    .bar span {{ display: block; height: 100%; border-radius: inherit; }}
    .notice {{
      border-left: 4px solid var(--amber);
      padding: 10px 12px;
      background: #fff7ed;
      color: #7a2e0e;
    }}
    .step-list {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      padding: 0;
      list-style: none;
    }}
    .step-list li {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fbfcff;
    }}
    footer {{ color: var(--muted); padding: 16px 2px 28px; font-size: 13px; }}
    @media (max-width: 860px) {{
      main {{ padding: 16px; }}
      .metrics, .grid, .step-list {{ grid-template-columns: 1fr; }}
      .metrics {{ margin-top: -28px; }}
      section {{ padding: 16px; }}
      table {{ display: block; overflow-x: auto; white-space: nowrap; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Analisis ambiental con Asyncio, MPI y CUDA</h1>
    <p>Proyecto de computo paralelo y distribuido sobre lecturas ThingSpeak y clima actual. El flujo obtiene datos concurrentemente, los reparte con MPI y aplica una fase acelerada por GPU para correlaciones y criticidad relativa.</p>
  </header>

  <main>
    <div class="metrics">
      {''.join(cards)}
    </div>

    <section>
      <h2>Resumen ejecutivo</h2>
      {fallback_html}
      <div class="grid">
        <div>
          <h3>ThingSpeak</h3>
          <table>
            <tr><th>Metrica</th><th>Valor</th></tr>
            <tr><td>Temperatura promedio</td><td>{_fmt(ts.get('temperatura_prom_global'), ' C')}</td></tr>
            <tr><td>Humedad promedio</td><td>{_fmt(ts.get('humedad_prom_global'), ' %')}</td></tr>
            <tr><td>Presion promedio</td><td>{_fmt(ts.get('presion_prom_global'), ' inHg')}</td></tr>
            <tr><td>Viento promedio</td><td>{_fmt(ts.get('velocidad_viento_prom_global'), ' mph')}</td></tr>
            <tr><td>Nivel de luz promedio</td><td>{_fmt(ts.get('nivel_luz_prom_global'))}</td></tr>
          </table>
        </div>
        <div>
          <h3>OpenWeather</h3>
          <table>
            <tr><th>Metrica</th><th>Valor</th></tr>
            <tr><td>Ciudad</td><td>{_e(weather.get('ciudad', 'N/A'))}</td></tr>
            <tr><td>Descripcion</td><td>{_e(weather.get('descripcion', 'N/A'))}</td></tr>
            <tr><td>Temperatura</td><td>{_fmt(weather.get('temperatura_c'), ' C')}</td></tr>
            <tr><td>Humedad</td><td>{_fmt(weather.get('humedad_pct'), ' %')}</td></tr>
            <tr><td>Adversidad ambiental</td><td>{_fmt(weather.get('adversidad_ambiental_idx'), '/100')}</td></tr>
          </table>
        </div>
      </div>
    </section>

    <section>
      <h2>Fase CUDA</h2>
      <p>Backend: <strong>{_e(backend)}</strong> | Dispositivo: <strong>{_e(device)}</strong> | Tiempo de calculo: <strong>{_fmt(cuda_meta.get('elapsed_ms'), ' ms', 3)}</strong></p>
      <h3>Estadisticas procesadas en matriz</h3>
      <table>
        <tr><th>Variable</th><th>Media</th><th>Desv. std</th><th>Min</th><th>Max</th></tr>
        {_stats_rows(cuda.get('estadisticas', {}))}
      </table>
    </section>

    <section>
      <h2>Criticidad relativa por proceso MPI</h2>
      <table>
        <tr><th>Bloque</th><th>Score</th><th>Valor</th><th>Nivel</th></tr>
        {_risk_rows(risk_rows)}
      </table>
    </section>

    <section>
      <h2>Matriz de correlacion CUDA</h2>
      {_corr_table(cuda.get('correlacion', {}))}
    </section>

    <section>
      <h2>Interpretacion</h2>
      {_interpretation(cuda.get('interpretacion', []))}
      <h3>Correlacion entre fuentes</h3>
      <p>{_e(corr_sources.get('interpretacion_ambiental', 'Sin interpretacion disponible.'))}</p>
      <p>Concordancia de temperatura: <strong>{_e(corr_sources.get('concordancia_temperatura', 'N/A'))}</strong></p>
    </section>

    <section>
      <h2>Ejecucion paso a paso</h2>
      <ol class="step-list">
        <li><strong>1. Asyncio</strong><br>Se usa <code>asyncio.gather</code> para consultar ThingSpeak y OpenWeather de forma concurrente.</li>
        <li><strong>2. MPI</strong><br><code>mpiexec</code> divide las lecturas por rank, ejecuta estadistica local y reune resultados con <code>gather</code>.</li>
        <li><strong>3. CUDA</strong><br>CuPy mueve la matriz de metricas a GPU, calcula z-scores, correlaciones y un kernel de criticidad.</li>
      </ol>
    </section>

    <footer>
      Generado: {_e(generated_at)} | {source_note}
    </footer>
  </main>
</body>
</html>
"""

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        file.write(html_doc)

    print(f"Reporte HTML generado en: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera el reporte HTML final.")
    parser.add_argument("--mpi", help="Ruta opcional al reporte_mpi_*.json.")
    parser.add_argument("--cuda", help="Ruta opcional al reporte_cuda_*.json.")
    parser.add_argument("--output", help="Ruta opcional del HTML de salida.")
    args = parser.parse_args()
    generate_html_report(args.mpi, args.cuda, args.output)


if __name__ == "__main__":
    main()
