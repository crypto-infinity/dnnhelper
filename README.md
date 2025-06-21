# dnnhelper

Modulo helper per Deep Neural Networks (DNN) con PyTorch.
Consente il training di reti neurali artificiali con PyTorch, a scopo didattico.

## Installazione

E' sufficiente installare le dipendenze:

```bash
pip install -r requirements.txt
```

## Requisiti PyPI
- torch
- torchmetrics
- scikit-learn
- numpy
- matplotlib
- pandas
- seaborn

## Descrizione
Questo modulo fornisce classi e funzioni per semplificare la creazione, l'addestramento, la validazione e la valutazione di reti neurali profonde in PyTorch, con supporto per metriche, early stopping, cross-validation e visualizzazione.

## Classi principali

### EarlyStopping
Implementa l'early stopping per interrompere l'addestramento quando la loss di validazione non migliora.

**Costruttore:**
```python
EarlyStopping(save_path, patience=5, min_delta=0.0)
```
- `save_path`: percorso dove salvare il checkpoint
- `patience`: epoche di tolleranza senza miglioramento
- `min_delta`: miglioramento minimo considerato

**Uso:**
```python
es = EarlyStopping('model.pt', patience=10, min_delta=0.01)
for epoch in range(epochs):
    ...
    es(val_loss, model)
    if es.early_stop:
        break
```

---

### Experiment
Gestisce la configurazione e la storia di un esperimento di training.

**Costruttore:**
```python
Experiment(
    name,
    checkpoints_folder,
    checkpoint_name,
    model,
    metrics,
    n_classes,
    loss_fn,
    optimizer,
    lr=1e-5,
    lr_scheduler=False,
    lr_gamma=0.1,
    lr_step=5,
    use_early_stopping=False,
    patience=5,
    min_delta=0.0,
    epochs=50
)
```
- `model`: istanza di torch.nn.Module
- `loss_fn`: classe loss PyTorch (es. nn.CrossEntropyLoss)
- `optimizer`: classe ottimizzatore PyTorch (es. optim.Adam)
- `metrics`: lista di metriche ('accuracy', 'precision', 'recall', 'f1')
- `n_classes`: numero di classi

**Esempio:**
```python
exp = Experiment(
    name='prova',
    checkpoints_folder='./checkpoints',
    checkpoint_name='best.pt',
    model=MyNet(),
    metrics=['accuracy', 'f1'],
    n_classes=10,
    loss_fn=nn.CrossEntropyLoss,
    optimizer=optim.Adam
)
```

---

### Helper
Funzioni statiche di utilità per visualizzazione e riproducibilità.

- `plot_images(dataset, classes, ...)` — Visualizza batch di immagini
- `plot_class_distribution(dataset, type="training")` — Istogramma distribuzione classi
- `plot_loss(exp)` — Loss training/validation
- `plot_confusion_matrix(y_true, y_pred, classes)` — Matrice di confusione
- `set_seed(seed)` — Imposta seed random
- `set_device()` — Restituisce device torch

**Esempio:**
```python
Helper.plot_loss(exp)
Helper.plot_class_distribution(train_ds)
```

---

### Trainer
Gestisce training, valutazione e predizione.

- `Trainer.fit(exp, train_dl, val_dl)` — Addestra il modello
- `Trainer.evaluate(exp, test_dl)` — Valuta il modello
- `Trainer.predict(exp, test_dl)` — Predice le classi

**Esempio:**
```python
Trainer.fit(exp, train_dl, val_dl)
loss, acc, prec, f1, rec = Trainer.evaluate(exp, test_dl)
y_pred = Trainer.predict(exp, test_dl)
```

---

### CrossValidation
Gestisce la cross-validation K-Fold.

**Costruttore:**
```python
CrossValidation(
    experiments,
    train_ds,
    val_ds,
    n_splits,
    batch_size=128,
    shuffle=True,
    seed=None,
    verbose=True
)
```
- `experiments`: lista di oggetti Experiment
- `train_ds`: dataset di training
- `val_ds`: dataset di validazione (opzionale)
- `n_splits`: numero di fold

**Esempio:**
```python
cv = CrossValidation([exp], train_ds, None, n_splits=5)
results = cv.run()
```

---

## Esempio completo
```python
import torch
import torch.nn as nn
import torch.optim as optim
from dnnhelper import Experiment, Trainer, Helper

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)
    def forward(self, x):
        return self.fc(x)

exp = Experiment(
    name='mnist',
    checkpoints_folder='./ckpt',
    checkpoint_name='mnist.pt',
    model=MyNet(),
    metrics=['accuracy', 'f1'],
    n_classes=10,
    loss_fn=nn.CrossEntropyLoss,
    optimizer=optim.Adam
)

# train_dl, val_dl = ...
Trainer.fit(exp, train_dl, val_dl)
Helper.plot_loss(exp)
```

## Licenza
MIT