# Original User Request

## 2026-11-01T21:40:03Z

Investigar, diseñar e implementar una solución reproducible para SemEval-2027 Task 2 (DiCo-NLI) en Track 1 (Inglés) y Track 2 (Español), que resuelva el Reversal Curse mediante regularización simétrica y decodificación consistente, con arquitectura desacoplada para validación local rápida y scripts CLI autocontenidos para entrenamiento pesado en servidor remoto con GPU NVIDIA (CUDA), incluyendo reporte en formato ACL y diapositivas de presentación en PDF.

Working directory: /Users/emilianodesu/Documents/LCD/7Sem/PLN/FinalProject
Integrity mode: development

## Requirements

### R1. Pipeline de Datos y Métricas de Consistencia Direccional (Track 1 y 2)
Implementar un módulo de carga, preprocesamiento y auditoría de datos para Track 1 (Inglés) y Track 2 (Español) del corpus PhrasIS de DiCo-NLI. El módulo debe:
1. Emparejar automáticamente cada par de frases $(A, B)$ con su par inverso $(B, A)$ y aplicar la inversión determinista de etiquetas:
   - `EQUIVALENCE` $\leftrightarrow$ `EQUIVALENCE`
   - `FORWARD_ENTAILMENT` $\leftrightarrow$ `BACKWARD_ENTAILMENT`
   - `BACKWARD_ENTAILMENT` $\leftrightarrow$ `FORWARD_ENTAILMENT`
   - `NEGATIVE_OTHER` $\leftrightarrow$ `NEGATIVE_OTHER`
2. Implementar y verificar programáticamente las tres métricas oficiales: `Weighted F1`, `SoftCons` (consistencia lógica entre pares independientes) y `HardCons` (ambas direcciones clasificadas correctamente).
3. Generar archivos de predicción que cumplan con el formato estricto de dos columnas de Codabench (`instance_id,label`).

### R2. Modelado y Estrategias contra el Reversal Curse
Desarrollar y comparar sistemáticamente tres familias de enfoques basados en Hugging Face Transformers (DeBERTa-v3 para inglés y BETO / RoBERTa-bne para español):
1. **Baseline Estándar:** Cross-encoder entrenado de forma naive unidireccional.
2. **Data Augmentation Simétrico (BSA):** Entrenamiento duplicado agregando la contraparte inversa $(B, A)$ con etiqueta revertida.
3. **Regularización Siamesa / Consistency Regularization Loss (CRL):** Arquitectura conjunta que procesa simultáneamente $(A, B)$ y $(B, A)$ penalizando divergencias de distribución mediante divergencia KL o Jensen-Shannon.
4. **Decodificación Simetrizada en Inferencia (SD):** Proyección post-hoc de probabilidades directas e inversas que garantiza teórica y empíricamente un $\text{SoftCons} = 100\%$.

### R3. Scripts CLI Autocontenidos para Servidor con GPU NVIDIA (CUDA)
Estructurar el código de manera modular y crear scripts de entrenamiento por línea de comandos (`experiments/train.py` y `scripts/run_gpu_training.sh`) totalmente reproducibles y autocontenidos:
1. Detección automática y óptima de dispositivo (`torch.cuda.is_available()`), mixed precision (`fp16`/`bf16`) y batch sizing adaptativo.
2. Guardado estructurado de checkpoints (`checkpoints/`), logs de métricas por época (`logs/`) y exportación automática de archivos de predicción en formato oficial (`results/`).
3. Flujo preparado para que el usuario simplemente clone el repositorio en su servidor GPU, ejecute un comando único (`bash scripts/run_gpu_training.sh`), y obtenga todos los modelos entrenados y métricas calculadas.

### R4. Estudio Experimental, Análisis de Errores y Ablaciones
Ejecutar experimentos de evaluación en los splits de desarrollo y validación para Track 1 y Track 2, produciendo:
1. Tabla comparativa de resultados (`Weighted F1`, `SoftCons`, `HardCons`) entre todos los métodos.
2. Matrices de confusión y matrices de transición de inconsistencia direccional.
3. Análisis cualitativo y cuantitativo de errores según la categoría sintáctica de las frases (sintagmas nominales, cadenas verbales, sintagmas preposicionales, adverbiales).

### R5. Entregables Académicos (Reporte ACL y Diapositivas PDF)
1. **Reporte Final en LaTeX (formato ACL):** Documento `report/paper.tex` compilable a PDF siguiendo la plantilla de SemEval 2027 (`semeval.github.io/system-paper-template.html`), conteniendo: Título, Autores, Abstract, Introducción, Descripción del Corpus, Metodología (formulaciones matemáticas rigurosas), Resultados, Análisis de Errores, Conclusiones y Referencias bibliográficas.
2. **Presentación Final:** Presentación en formato PDF (`presentation/presentation.pdf`) que contenga: Tabla de contenido, Introducción, Metodología, Descripción del corpus, Resultados y Conclusiones.

## Acceptance Criteria

### Verificación Automatizada y Tests
- [ ] Suite de pruebas en `tests/test_reversal_logic.py` y `tests/test_metrics.py` ejecutada y aprobada con 100% de éxito.
- [ ] Test unitario demostrando que la Decodificación Simetrizada (SD) produce matemáticamente $\text{SoftCons} = 1.0$ en cualquier matriz de predicciones.
- [ ] Validación automática de formato de salida Codabench (`instance_id,label`) sin filas vacías ni valores nulos.
- [ ] Pipeline de prueba rápida de extremo a extremo (`tests/test_pipeline.py`) ejecutable localmente en CPU en menos de 60 segundos con datos mock.

### Facilidad de Ejecución en Servidor GPU
- [ ] Script `scripts/run_gpu_training.sh` y `experiments/train.py` probados y documentados en el `README.md` con flags claros (`--track`, `--model`, `--method`, `--batch_size`, `--fp16`).
- [ ] Mecanismo de exportación de predicciones y métricas listo para ser enviado a Codabench o recuperado del servidor.

### Calidad de Entregables
- [ ] `report/paper.pdf` generado y libre de errores de compilación, cumpliendo las 8 secciones mínimas de la rúbrica.
- [ ] `presentation/presentation.pdf` generado cumpliendo los 6 puntos de la rúbrica de exposición.
- [ ] Repositorio Git limpio, modular y con instrucciones en `README.md`.
