# Tarea 2: Tuning de SVM y MLP

Trabajo práctico de la materia **Ingeniería de atributos y modelos para ML** (FAMAF, Universidad Nacional de Córdoba, 2024).

El objetivo es predecir si un préstamo de Lending Club **no será pagado en término** (`not.fully.paid`), ajustando los hiperparámetros de una SVM con kernel RBF y de un perceptrón multicapa (MLP). Después se evalúan los modelos con una métrica de negocio: el retorno por préstamo (ROI) de un prestamista.

## Contenido

| Archivo | Descripción |
|---|---|
| `TP2_SVM_MLP.ipynb` | Notebook con todo el desarrollo, los resultados y las interpretaciones |
| `loan_data.csv` | Dataset *Loan Data* de Lending Club (9578 préstamos, 14 atributos) |

## Desarrollo

1. **Preparación:** codificación de `purpose` con `get_dummies`, separación train/test estratificada (67/33) y estandarización.
2. **SVM con kernel RBF:** búsqueda de `C` y `gamma` con `GridSearchCV`, optimizando el F1 de la clase *no pagó*.
3. **SVM con clases balanceadas:** submuestreo de la clase mayoritaria (1000 ejemplos) y selección de 4 atributos (`credit.policy`, `int.rate`, `fico`, `inq.last.6mths`) según su correlación con la clase.
4. **MLP:** selección del tamaño de una y dos capas ocultas según `best_loss_`, y diagnóstico de sobreajuste.
5. **ROI por préstamo:** comparación de todos los modelos con los baselines *random* y *majority*, y con Naive Bayes.
6. **Discusión** de los resultados desde el dominio.

## Resultados principales

ROI por préstamo sobre el test set, con beneficio 0,3 por préstamo pagado, costo 0,8 por préstamo impago y costo de oportunidad 0,3 por buen pagador rechazado:

| Modelo | Morosidad de la cartera | ROI por préstamo |
|---|---|---|
| Majority (presta a todos) | 16,0 % | **0,124** |
| SVM | 11,7 % | 0,079 |
| Naive Bayes (4 atributos, balanceado) | 12,2 % | 0,074 |
| MLP (220, 60) | 13,5 % | −0,038 |
| Random | 16,9 % | −0,145 |
| SVM (4 atributos, balanceado) | 8,9 % | −0,403 |

- **Ningún clasificador supera a prestarle a todos los solicitantes.** Los modelos reducen la morosidad y aumentan la ganancia por peso prestado, pero rechazan demasiados buenos pagadores.
- **Los atributos tienen poco poder predictivo:** ninguno supera una correlación de 0,17 con la clase, y el mejor balanced accuracy sobre el test es de alrededor de 0,60.
- **La MLP sobreajusta:** elegir la arquitectura por `best_loss_` (pérdida sobre el train) lleva a la red más grande, que alcanza 0,98 de balanced accuracy en train y 0,55 en test.
- **Origen del problema:** el dataset sólo contiene préstamos ya aprobados por Lending Club (sesgo de selección), y su evaluación de riesgo ya está incorporada en la tasa de interés.

## Cómo ejecutarlo

Requiere Python 3 (probado con Python 3.12, numpy 2.0.2, pandas 2.2.2, matplotlib 3.9.2 y scikit-learn 1.5.2).

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook TP2_SVM_MLP.ipynb
```

El notebook lee `loan_data.csv` desde su misma carpeta. La ejecución completa tarda unos 6 minutos, principalmente por la búsqueda en grilla de la SVM.

## Créditos

- Enunciado: P. Pury, FaMAF 2024 (CC BY-NC-SA).
- Dataset: *Loan Data* de Lending Club, disponible en Kaggle.
