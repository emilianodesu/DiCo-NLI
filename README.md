# SemEval-2027 Task 2: DiCo-NLI (Proyecto Final PLN)

Repositorio oficial del proyecto de final de curso de **Procesamiento de Lenguaje Natural (PLN)** para la tarea compartida **SemEval-2027 Task 2: DiCo-NLI** (*Directional Consistency in Fine-Grained Natural Language Inference*).

## Descripción del Proyecto
Este proyecto aborda el fenómeno del *Reversal Curse* y la inconsistencia lógica direccional en inferencia de lenguaje natural a nivel de frase (corpus PhrasIS). Evaluamos y proponemos soluciones para:
* **Track 1:** Inglés (monolingüe)
* **Track 2:** Español (monolingüe)

### Métricas de Evaluación
* **Weighted F1:** Rendimiento de clasificación multiclase individual.
* **SoftCons:** Tasa de consistencia lógica entre predicciones inversas $(A, B)$ y $(B, A)$.
* **HardCons:** Tasa estricta donde **ambas** direcciones son clasificadas correctamente según la referencia.

### Métodos Investigados
1. **Baselines Unidireccionales:** Fine-tuning estándar de Transformers (DeBERTa-v3, BETO, RoBERTa).
2. **Data Augmentation Simétrico (BSA):** Aumento del conjunto de entrenamiento con pares invertidos y etiquetas transpuestas.
3. **Regularización de Consistencia Siamesa (CRL):** Entrenamiento con pérdida conjunta penalizando discrepancias distributivas mediante divergencia KL.
4. **Decodificación Simetrizada en Inferencia (SD):** Proyección equivariante post-hoc que garantiza $\text{SoftCons} = 100\%$.

## Estructura del Repositorio
```
├── data/                  # Datos crudos y procesados del benchmark
├── src/                   # Código fuente modular (data, models, training, evaluation)
├── experiments/           # Scripts reproducibles de entrenamiento y evaluación
├── notebooks/             # Notebooks para Google Colab con soporte GPU
├── tests/                 # Pruebas unitarias y de integración
├── report/                # Reporte final en formato ACL (LaTeX / PDF)
└── presentation/          # Diapositivas de la presentación en PDF
```

## Instalación y Configuración Local
```bash
# 1. Crear y activar entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar pruebas unitarias
pytest tests/
```

## Entrenamientos en GPU (Google Colab / Kaggle)
Consulta el notebook autónomo ubicado en `notebooks/DiCo_NLI_Colab_Training.ipynb` para ejecutar los experimentos en entornos con GPU NVIDIA.
