# uncertainty-aware-moderation

```python -m src.gaussian_processes.train --freeze_encoder```

entrena el setup tipo “GP only” en el sentido práctico de tu proyecto: el encoder queda congelado y solo se entrenan la proyección lineal, las 6 GP heads y sus likelihoods. No es “solo GP” puro porque las features siguen viniendo de DistilBERT, pero DistilBERT no se actualiza.

```python -m src.gaussian_processes.train```

entrena el modelo completo con GPs y fine-tuning conjunto: se actualiza el encoder de DistilBERT y también la proyección, las GP heads y las likelihoods.

```python -m src.finetuning.train```

sería el baseline clásico de fine-tuning: encoder + classifier head normal, sin GP.

## Deploy the web
In one terminal:

```python -m uvicorn src.web.api:app --reload```

In another:
```python -m streamlit run src/web/app.py```

