from sklearn.metrics import cohen_kappa_score
from autogluon.core.metrics import make_scorer

def kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred)

custom_kappa_metric = make_scorer(name='cohens_kappa',
                    score_func=kappa,
                    optimum=1,
                    greater_is_better=True)