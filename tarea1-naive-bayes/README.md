# Selección de atributos con Naive Bayes: predicción de préstamos impagos

Trabajo práctico N°1 del curso **Ingeniería de Atributos y Modelos de Machine Learning**.

## Objetivo

Predecir si una persona **no va a devolver completamente un préstamo** (`not.fully.paid = 1`) y encontrar el **conjunto más chico de atributos** que mantenga el F1-score casi igual y lleve el *recall* de la clase minoritaria por lo menos a 0.2.

## Dataset

[Loan Data](https://www.kaggle.com/datasets/itssuru/loan-data) (Kaggle), con datos de préstamos de LendingClub.

- 9.578 préstamos y 13 atributos (tasa de interés, cuota, puntaje FICO, ingresos, endeudamiento, etc.)
- Clases desbalanceadas: **84 %** pagaron el préstamo y **16 %** no lo pagaron.

> El dataset no está incluido en el repositorio. Para correr el notebook, descargá `loan_data.csv` desde Kaggle y guardalo en la misma carpeta.

## Metodología

1. **Preprocesamiento:** *one-hot encoding* de la variable categórica `purpose`.
2. **Train/validación:** separación 67/33 estratificada (`random_state=125`).
3. **Modelo base:** `GaussianNB` de scikit-learn.
4. **Balanceo de clases:** *undersampling* de la clase mayoritaria, **solo en el set de entrenamiento**, para que la validación refleje la distribución real.
5. **Análisis de atributos:** se eliminó un atributo por vez y se registró cómo cambiaban las métricas.
6. **Selección hacia atrás manual:** se fueron eliminando atributos y se conservaron los que mejoraban el *recall* sin perder mucho *accuracy* ni F1.

## Resultados

| Modelo | Atributos | Accuracy | Recall (clase 1) | Precision (clase 1) | F1 (weighted) |
|---|---|---|---|---|---|
| Naive Bayes sin balancear | 19 | 0.82 | 0.08 | 0.27 | 0.77 |
| Naive Bayes balanceado | 19 | 0.78 | 0.26 | 0.28 | 0.77 |
| **Balanceado + selección de atributos** | **3** | **0.75** | **0.39** | **0.28** | **0.76** |

Con solo **3 atributos** (`credit.policy`, `installment` e `inq.last.6mths`), el *recall* de la clase de interés sube de 0.08 a 0.39 y el F1 casi no cambia.

## Conclusiones

- Con clases desbalanceadas, el *accuracy* engaña: el modelo inicial acertaba el 82 % pero detectaba solo el 8 % de los impagos.
- Balancear el entrenamiento mejora mucho la detección de la clase minoritaria.
- Muchos atributos aportan información redundante: un modelo más simple puede rendir igual o mejor.

## Tecnologías

Python · pandas · NumPy · scikit-learn · Matplotlib · Google Colab

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
jupyter notebook Bencharski_Tarea1.ipynb
```

O abrilo directamente en Google Colab.
