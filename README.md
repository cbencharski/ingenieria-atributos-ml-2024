# Ingeniería de Atributos y Modelos de Machine Learning

Trabajos prácticos entregables del curso **Ingeniería de Atributos y Modelos de Machine Learning** (FaMAF, Universidad Nacional de Córdoba, 2024).
Cada carpeta contiene un proyecto independiente con su notebook y un README que explica el problema, la metodología y los resultados.

Todos los trabajos usan el dataset [Loan Data](https://www.kaggle.com/datasets/itssuru/loan-data) de Lending Club, donde el objetivo es predecir si un préstamo **no será devuelto por completo**.

## Proyectos

| # | Proyecto | Temas | Modelo |
|---|---|---|---|
| 1 | [Selección de atributos: predicción de préstamos impagos](./tarea1-naive-bayes) | Clases desbalanceadas, *undersampling*, selección de atributos, métricas de clasificación | Naive Bayes |
| 2 | [Tuning de SVM y MLP: retorno de inversión en préstamos](./tarea2-SVM-MLP) | Búsqueda de hiperparámetros (GridSearchCV), redes neuronales, sobreajuste, métrica de negocio (ROI) | SVM (RBF), MLP |
| 3 | *Próximamente* | | |

## Tecnologías

Python · pandas · NumPy · scikit-learn · Matplotlib · Google Colab / Jupyter

## Estructura del repositorio

```
ingenieria-atributos-ml/
├── README.md
├── tarea1-naive-bayes/
│   ├── README.md
│   ├── Bencharski_Tarea1.ipynb
│   └── requirements.txt
└── tarea2-SVM-MLP/
    ├── README.md
    ├── TP2_SVM_MLP.ipynb
    └── loan_data.csv
```

## Autora

**Constanza Bencharski
