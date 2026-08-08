

<div align="center">

# 🔋 Pronóstico de Baterías Blindado por Física

### PINN Desacoplado de Escala de Tiempo Micro-Macro para el Pronóstico de Baterías Crítico para la Seguridad

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/Zhi-Chao-PAN/safety-critical-battery-prognostics/actions/workflows/ci.yml/badge.svg)](https://github.com/Zhi-Chao-PAN/safety-critical-battery-prognostics/actions/workflows/ci.yml)
[![Synthetic VR](https://img.shields.io/badge/Synthetic%20VR-0.00%25%20(3--Layer%20Defense)-success.svg)](docs/comprehensive_experimental_results.md)
[![GitHub stars](https://img.shields.io/github/stars/Zhi-Chao-PAN/safety-critical-battery-prognostics?style=social)](https://github.com/Zhi-Chao-PAN/safety-critical-battery-prognostics)

*Una defensa de física de tres capas con **tasa de violación física del 0.00%**¹ en la prueba de robustez sintética y evidencia acotada y equilibrada por equidad en datos reales de 6 celdas CALCE.*

<sub>¹ El titular del 0.00% VR específico del modelo proviene de la prueba sintética `robustness_test.py`. Los informes de datos reales de misma celda, LOGO y corrupción multi-semilla aplican exactamententico suavizado EMA + proyección monótona a PINN y baseline; en esos protocolos equilibrados por equidad, tanto PINN como LSTM alcanzan 0.00% VR mientras PINN se retrasa en RMSE. Consulte [Resultados Completos](docs/comprehensive_experimental_results.md) y la [Matriz de Afirmaciones-Evidencia](docs/claim_evidence_matrix.md) para los límites del protocolo.</sub>

[📄 Borrador del Artículo](docs/archive/IEEE_Whitepaper_PINN_Battery_RUL_Complete.md) · [📊 Resultados Completos](docs/comprehensive_experimental_results.md) · [🇨🇳 简体中文](README_zh.md)

> **Nota del repositorio**: las fronteras de afirmaciones autorizadas y actualizadas para este repositorio residen en este `README`, [Resultados Completos](docs/comprehensive_experimental_results.md) y la [Matriz de Afirmaciones-Evidencia](docs/claim_evidence_matrix.md). Los archivos bajo `docs/archive/` se conservan como entregables históricos.

</div>

---

## 🎯 El Problema

En los Sistemas de Gestión de Baterías (BMS) críticos para la seguridad, una sola **predicción no física** — como pronosticar un aumento de capacidad durante el envejecimiento de la batería — puede desencadenar fallos catastróficos: estimaciones de autonomía erróneas en vehículos eléctricos, apagados prematuros de almacenamiento en red eléctrica o advertencias perdidas de fuga térmica.

Los modelos puramente impulsados por datos (LSTM, Transformer) suelen rutinariamente este tipo de violaciones bajo ruido sensorial. **Este repositorio muestra cómo una defensa de física de tres capas las elimina en la prueba de robustez sintética, manteniendo las afirmaciones en datos reales explícitamente acotadas y específicas del protocolo.**

---

## 🛡️ Arquitectura de la Defensa de Física de Tres Capas

Nuestra contribución principal es un **escudo de física en cascada** que garantiza predicciones de degradación de capacidad monótona en la prueba de robustez sintética bajo un 50% de ruido gaussiano:

```
┌─────────────────────────────────────────────────────────┐
│  Capa 1: Entrenamiento con Restricciones                │
│  → Incorpora el priors de física mediante penalización  │
│    diferenciable                                      │
│  → Regularización suave durante la optimización         │
├─────────────────────────────────────────────────────────┤
│  Capa 2: Ajuste de Límite del Residual                  │
│  → Limita el residual de la NN al rango observado       │
│    en el entrenamiento                                  │
│  → Previene explosión OOD (RMSE ↓77%)                  │
├─────────────────────────────────────────────────────────┤
│  Capa 3: Proyección Monótona                            │
│  → Suavizado EMA (α=0.15) + mínimo en tiempo real     │
│  → Garantía estricta: 0.00% de violaciones físicas      │
└─────────────────────────────────────────────────────────┘
```

### ¿Por qué tres capas?

Cada capa aborda un **modo de fallo distinto**. Nuestro estudio de ablación demuestra que son complementarias y no redundantes:

| Configuración de Defensa | RMSE (Ah) | Tasa de Violación | Rol |
|--------------------------|-----------|-------------------|-----|
| Sin Defensa            | 1.748     | 50.75%            | Línea base (catastrófica) |
| + Entrenamiento con Restricciones | 3.348 | 48.24%            | Regularización débil |
| + Ajuste de Límite del Residual | **0.759** | 48.74%            | **Precisión** (RMSE ↓77%) |
| + Proyección Monótona  | 2.589     | **0.00%**         | Garantía de **Seguridad** |
| **Defensa Completa (Nuestra)** | **0.323** | **0.00%**       | **Lo mejor de ambos** |

> **Esencia Clave**: Eliminar el ajuste de límite degrada la precisión (RMSE 2.589). Eliminar la proyección rompe la seguridad (VR ~48%). Se requieren las tres.

---

## 📊 Resultados Experimentales

### Robustez: PINN vs LSTM bajo un 50% de ruido gaussiano

| Métrica | PINN (Nuestro) | LSTM Baseline |
|---------|:--------------:|:-------------:|
| Tasa de Violación Física | **0.00%** ✅ | 18.55% ❌ |
| Latencia de Inferencia | **11 ms** ⚡ | 2,230 ms |
| Ventaja de Velocidad | **203× más rápido** | — |

### Robustez al Ruido en la Misma Celda (Emparejado por Equidad)

La reejecución con semilla de `scripts/validate_real_data.py` mantiene el protocolo acotado a la rechazo de ruido en la misma celda y aplica la misma cadena de postprocesamiento a ambos modelos:

| Condición | RMSE Promedio PINN | VR Promedio PINN | RMSE Promedio LSTM | VR Promedio LSTM |
|-----------|--------------------|------------------|--------------------|------------------|
| Trayectoria de misma celda con 50% de ruido | 0.3848 | 0.00% | 0.2160 | 0.00% |

> **Interpretación acotada**: con el mismo suavizado EMA + proyección de mínimo en tiempo real, tanto PINN como LSTM son monótonos en las 6 celdas reales bajo este protocolo de misma celda. Esto ya no constituye evidencia de una ventaja de seguridad específica de PINN en datos reales, y CS2_36 es el pliegue más difícil para PINN (RMSE 1.1494).

### Validación Cruzada entre Celdas LOGO (Celdas Retenidas, Conclusión Acotada)

El repositorio ahora incluye una validación ejecutada de "dejar una celda fuera" en las mismas 6 celdas CALCE:

```bash
python scripts/validate_real_data_logo.py
```

Este protocolo entrena con todas las celdas limpias no retenidas y evalúa en la celda retenida bajo condiciones limpias y ruidosas. Es la ruta correcta para la evidencia real cruzada entre celdas y se mantiene intencionalmente separada de la tabla de robustez al ruido en la misma celda anterior.

| Condición | RMSE Promedio PINN | VR Promedio PINN | RMSE Promedio LSTM | VR Promedio LSTM |
|-----------|--------------------|------------------|--------------------|------------------|
| Celda retenida limpia | 0.2497 | 0.00% | 0.2223 | 0.00% |
| Celda retenida con 50% de ruido | 0.2615 | 0.00% | 0.2232 | 0.00% |

> **Interpretación acotada**: la reejecución con semilla del LOGO muestra nuevamente que tanto PINN como LSTM se mantienen en una tasa de violación del 0.00% bajo la pila de postprocesamiento compartida en celdas retenidas, mientras que PINN sigue por detrás de LSTM en RMSE. El pliegue más difícil de PINN ahora es CS2_33 en lugar de un modo de fallo uniforme en todas las celdas.

### Suite de Estrés de Corrupción con Múltiples Semillas

El repositorio ahora también incluye un reporte de suite de estrés con semillassemillas fijas a través de 5 semillas de corrupción y 4 familias de corrupción para los protocolos de misma celda y LOGO:

| Protocolo | Rango de RMSE de PINN a través de Corrupciones | Rango de RMSE de LSTM a través de Corrupciones | VR Compartida |
|-----------|------------------------------------------------|------------------------------------------------|---------------|
| Misma celda | 0.3941-0.4012 | 0.2158-0.2160 | 0.00% para ambos |
| LOGO | 0.2499-0.2572 | 0.2224-0.2226 | 0.00% para ambos |

Consulte [real_data_stress_suite_report.md](robustness_results/real_data_stress_suite_report.md) para las tablas `media ± desv. estándar` por corrupción y los desgloses de pliegue más difícil.

### Eficiencia Computacional

| Métrica | Valor |
|---------|-------|
| VRAM Pico | 8.14 MB |
| Inferencia ONNX INT8 | < 0.1 ms |
| Aceleración Entrenamiento AMP | 2× (Tensor Core) |
| Aceleración MC Dropout | 100× (Lote) |

---

## 🏗️ Arquitectura

```
                    ┌──────────────────────┐
                    │    Datos de Batería   │
                    │  (V, I, T, ciclos)    │
                    └─────────┬────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼                               ▼
   ┌──────────────────┐            ┌──────────────────┐
   │ SPM Micro-Escala  │           │ NN Macro-Escala  │
   │ (Intra-ciclo)     │           │ (Inter-ciclo)    │
   │                    │           │                    │
   │  DifDifusión de Fick  │──feat──▶ │ TCN + AtenciónAtención    │
   │  Sandbox FDM       │           │ λ(t) Adaptativo  │
   └──────────────────┘            └────────┬─────────┘
                                             │
                              ┌──────────────┼──────────────┐
                              ▼              ▼              ▼
                        ┌──────────┐  ┌──────────┐  ┌──────────┐
                        │ Capa 1   │  │ Capa 2   │  │ Capa 3   │
                        │ Restric. │→ │ Ajuste   │→ │ Proyect. │
                        │ Entrenam.│  │ Residual │  │ Monótono │
                        └──────────┘  └──────────┘  └──────────┘
                                             │
                                             ▼
                                    ┌────────────────┐
                                    │  Predicción RUL │
                                    │  0.00% VR ✅    │
                                    └────────────────┘
```

### Innovaciones Clave

1. **Desacoplamiento de Escala de Tiempo Micro-Macro** — Resuelve el "agujero negro de escala temporal" aislando la dinámica rápida de SPM (segundos) de la predicción lenta de degradación (meses)
2. **Ponderación Adaptativa de la Pérdida Física** — λ(t) programado con sigmoide confía en los datos al inicio y en la física al final
3. **Escudo de Física de Tres Capas** — Defensa en cascada con 0% de violaciones físicas en la prueba sintética
4. **MC Dropout en Lotes** — Cuantificación de incertidumbre 100× más rápida mediante expansión de tensor
5. **Entrenamiento AMP** — Aceleración 2× con reducción del 41% de VRAM en RTX 4060

---

## 🚀 Inicio Rápido

```bash
# Clonar
git clone https://github.com/Zhi-Chao-PAN/safety-critical-battery-prognostics.git
cd safety-critical-battery-prognostics

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar demostración básica
python main.py

# Ejecutar prueba de robustez (PINN vs LSTM bajo 50% de ruido)
python robustness_test.py

# Ejecutar estudio de ablación de capas de defensa
python scripts/ablation_defense_layers.py

# Ejecutar validación real de CALCE
python scripts/validate_real_data.py

# Ejecutar validación cruzada entre celdas LOGO real
python scripts/validate_real_data_logo.py

# Ejecutar pruebas unitarias
python -m pytest tests/ -v
```

---

## 📁 Estructura del Repositorio

```text
safety-critical-battery-prognostics/
├── src/                        # Código fuente principal
│   ├── models/                 #   PINN, LSTM, Chronos, Adaptador en Línea
│   ├── physics/                #   SPM diferenciable, sistema de restricciones
│   ├── training/               #   Precisión mixta, LOGO-CV
│   ├── data/                   #   Ingestión y normalización de datos
│   ├── evaluation/             #   Métricas y perfilado de rendimiento
│   ├── uncertainty/            #   Predicción conformal, MC Dropout
│   ├── safety/                 #   Motor de diagnóstico LLM-FMEA
│   ├── deployment/             #   Exportación ONNX y canalización de cuantización
│   └── infrastructure/         #   Esquema de configuración, gestión de conjuntos de datos
├── scripts/                    # Scripts de experimentos
│   ├── ablation_defense_layers.py  # Ablación de capas de defensa (5 variantes)
│   ├── validate_real_data.py       # Validación robusta en misma celda (ruido)
│   ├── validate_real_data_logo.py  # Validación cruzada entre celdas LOGO
│   ├── validate_real_data_stress_suite.py # Suite de corrupción con múltiples semillas
│   ├── run_ablation_study.py       # Ablación de arquitectura
│   └── ...
├── robustness_results/         # Resultados de todos los experimentos de robustez
│   ├── ablation_defense_layers.png # Figura de ablación de grado IEEE
│   ├── real_data_validation.png    # Figura de ruido en misma celda de 12 paneles
│   ├── real_data_logo_validation.png # Figura LOGO limpia/ruidosa
│   ├── real_data_logo_validation_report.md # Resumen markdown LOGO
│   ├── real_data_stress_suite_report.md # Reporte de corrupción con múltiples semillas
│   └── *.md, *.csv                 # Reportes y datos sin procesar
├── data/                       # Conjuntos de datos NASA + CALCE
├── tests/                      # 91 pruebas automatizadas (100% exitosas)
├── docs/                       # Documentación y borrador de artículo
├── configs/                    # Configuraciones YAML (esquema + experimentos)
└── robustness_test.py          # Canalización principal de robustez
```

---

## 📄 Documentación

| Documento | Descripción |
|-----------|-------------|
| [Resultados Completos](docs/comprehensive_experimental_results.md) | Informe experimental completo |
| [Matriz de Afirmaciones-Evidencia](docs/claim_evidence_matrix.md) | Afirmaciones verificadas, afirmaciones acotadas y trabajo futuro |
| [Guía de Contribución](CONTRIBUTING.md) | Cómo configurar un entorno de desarrollo, ejecutar verificaciones y enviar PRs |
| [Código de Conducta](CODE_OF_CONDUCT.md) | Expectativas comunitarias para una colaboración respetuosa |
| [Política de Seguridad](SECURITY.md) | Cómo reportar preocupaciones de seguridad, seguridad operativadocumentación o uso indebido |
| [Progreso del Proyecto](docs/project_progress.md) | 16 hitos con métricas |
| [Borrador del Artículo IEEE](docs/archive/IEEE_Whitepaper_PINN_Battery_RUL_Complete.md) | Artículo completo estilo IEEE Transactions |
| [Guía de Arquitectura](docs/PROJECT_ARCHITECTURE.md) | Documentación de diseño del sistema |
| [Guía de Despliegue](docs/deployment/DEPLOYMENT_GUIDE.md) | SOP de despliegue en BMS de borde |

Las fuentes de verdad activas son las primeras cuatro entradas anteriores. Los materiales del archivo permanecen disponibles para la procedencia, pero no deben usarse como fuente principal para las afirmaciones actuales del conjunto de referencia.

---

## 🧪 Reproducibilidad

Todos los experimentos son completamente reproducibles con semillas aleatorias fijas:

```bash
# Repetir ablación de defensa (Tabla III en el artículo)
python scripts/ablation_defense_layers.py
# → robustness_results/ablation_defense_layers.png
# → robustness_results/ablation_defense_report.md

# Repetir validación de ruido en misma celda (Tabla IV en el artículo)
python scripts/validate_real_data.py
# → robustness_results/real_data_validation.png
# → robustness_results/real_data_validation_report.md

# Repetir validación cruzada entre celdas LOGO
python scripts/validate_real_data_logo.py
# → robustness_results/real_data_logo_validation.png
# → robustness_results/real_data_logo_validation_report.md

# Repetir la suite de estrés de corrupción con múltiples semillas
python scripts/validate_real_data_stress_suite.py
# → robustness_results/real_data_stress_suite_report.md
# → robustness_results/real_data_stress_suite_summary.csv
```

---

## 📄 Cita

```bibtex
@software{pan2026pinn_battery,
  author = {Pan, Zhichao},
  title = {Prognóstico de Baterías Blindado por Física: PINN Desacoplado de
           Escala de Tiempo Micro-Macro con Defensa de Tres Capas},
  year = {2026},
  url = {https://github.com/Zhi-Chao-PAN/safety-critical-battery-prognostics},
  note = {Tasa de violación física del 0.00\% en la prueba de robustez sintética; los informes de datos reales de misma celda y LOGO emparejados por equidad se incluyen por separado}
}
```

---

## 📬 Contacto

- **Autor**: Zhichao Pan
- **Correo electrónico**: [18652585856@163.com](mailto:18652585856@163.com)
- **GitHub**: [@Zhi-Chao-PAN](https://github.com/Zhi-Chao-PAN)

---

## 📄 Licencia

Licencia MIT — consulte [LICENSE](LICENSE) para más detalles.

---

<div align="center">

*Si este proyecto avanza su investigación, considere dejar una ⭐*

**0.00% de violaciones en la prueba sintética · inferencia 203× más rápida · informes de misma celda, LOGO y suite de estrés incluidos**

</div>
