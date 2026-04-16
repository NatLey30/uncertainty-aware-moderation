# Uncertainty aware moderation

We study the use of uncertainty aware models for multi-label classification of the comments made in a social media platform. The model flags the comments that have a negative component that could negatively affect the experience of other users. Our goal is to detect harmful comments, but also to estimate when the model is uncertain and human review may be necessary. The results show that probabilistic models can achieve good predictive performance while providing uncertainty that is related to the prediction errors. We also study the model's robustness against adversarial attacks and observe that the uncertainty estimation could serve as a base safety layer to detect toxic content.

## Use of the repository

### Train

- Train the GP only model, with the frozen encoder 

```python
python -m src.gaussian_processes.train --freeze_encoder
```

- Train the GP model and the encoder

```python
python -m src.gaussian_processes.train
```

- Train the GP model and the encoder with weights

```python
python -m src.gaussian_processes.train --use_weights
```

- Train the deterministic model (DistilBERT)

```python
python -m src.finetuning.train
```


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

```
Uncertainty-aware-moderation/
├── data/
├── images/
├── models/
├── outputs/
├── results_notebook/
│
├── src/
│   ├── finetuning/
│   │   ├── evaluate.py
│   │   ├── model.py 
│   │   ├── prediction.py 
│   │   ├── train_functions.py 
│   │   └── train.py
│   ├── gaussian_processes/
│   │   ├── evaluate.py
│   │   ├── model.py 
│   │   ├── prediction.py 
│   │   ├── train_functions.py 
│   │   └── train.py
│   ├── web/
│   │   ├── api.py
│   │   ├── app.py 
│   │   └── plots.py
│   ├── data.py
│   └── utils.py
│
├── model_analysis_distilbert.ipynb
├── model_analysis_finetuned_weights.ipynb
├── model_analysis_finetuned.ipynb
├── model_analysis.ipynb
└── model_comparison.ipynb
```