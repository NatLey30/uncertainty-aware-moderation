# Uncertainty aware moderation

We study the use of uncertainty aware models for multi-label classification of the comments made in a social media platform. The model flags the comments that have a negative component that could negatively affect the experience of other users. Our goal is to detect harmful comments, but also to estimate when the model is uncertain and human review may be necessary. The results show that probabilistic models can achieve good predictive performance while providing uncertainty that is related to the prediction errors. We also study the model's robustness against adversarial attacks and observe that the uncertainty estimation could serve as a base safety layer to detect toxic content.

## Use of the repository

### Train

```python
python -m src.gaussian_processes.train --freeze_encoder
```

entrena el setup tipo “GP only” en el sentido práctico de tu proyecto: el encoder queda congelado y solo se entrenan la proyección lineal, las 6 GP heads y sus likelihoods. No es “solo GP” puro porque las features siguen viniendo de DistilBERT, pero DistilBERT no se actualiza.

```python
python -m src.gaussian_processes.train
```

entrena el modelo completo con GPs y fine-tuning conjunto: se actualiza el encoder de DistilBERT y también la proyección, las GP heads y las likelihoods.

```python
python -m src.finetuning.train
```

sería el baseline clásico de fine-tuning: encoder + classifier head normal, sin GP.

### Deploy the demo web
In one terminal:

```python
python -m uvicorn src.web.api:app --reload
```

In another:
```python
python -m streamlit run src/web/app.py
```


## Structure

