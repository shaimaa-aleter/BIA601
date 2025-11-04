import time
import numpy as np
from typing import Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from .config import CFG
from .utils import dynamic_cv


def ga_initialize(pop_size: int, n_features: int, init_prob: float = 0.3) -> np.ndarray:
    assert n_features > 0, "No features to initialize."
    return (np.random.rand(pop_size, n_features) < init_prob).astype(int)


def ga_tournament_select(pop: np.ndarray, fits: np.ndarray, k: int = 3) -> np.ndarray:
    idx = np.random.randint(0, len(pop), k)
    return pop[idx[np.argmax(fits[idx])]].copy()


def ga_uniform_crossover(p1: np.ndarray, p2: np.ndarray, prob: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    if np.random.rand() < prob:
        m = np.random.rand(len(p1)) < 0.5
        return np.where(m, p1, p2), np.where(m, p2, p1)
    return p1.copy(), p2.copy()


def ga_mutate(ind: np.ndarray, prob: float) -> np.ndarray:
    flip = np.random.rand(len(ind)) < prob
    ind[flip] = 1 - ind[flip]
    return ind


def ga_fitness(mask: np.ndarray, X: np.ndarray, y: np.ndarray, model, alpha: float = 0.01) -> float:
    assert mask.ndim == 1 and mask.shape[0] == X.shape[1], "Mask must be 1D over columns."
    if mask.sum() == 0:
        return 0.0
    Xs = X[:, mask == 1]
    n_splits = dynamic_cv(y, desired_splits=CFG.GA_CV)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CFG.RANDOM_STATE)
    scores = []
    for tr, va in skf.split(Xs, y):
        model.fit(Xs[tr], y[tr])
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(Xs[va])[:, 1]
            scores.append(roc_auc_score(y[va], p))
        else:
            scores.append(accuracy_score(y[va], model.predict(Xs[va])))
    score = float(np.mean(scores))
    penalty = alpha * (mask.sum() / len(mask))
    return score - penalty


def genetic_feature_selection_v2(
    X: np.ndarray,
    y: np.ndarray,
    model,
    pop_size: int = CFG.GA_POP,
    generations: int = CFG.GA_GENS,
    crossover_prob: float = CFG.GA_CROSSOVER,
    mutation_prob: float = CFG.GA_MUTATION,
    tournament_k: int = CFG.GA_TOURNAMENT_K,
    elitism: bool = CFG.GA_ELITISM,
    alpha: float = CFG.GA_ALPHA,
    patience: int = CFG.GA_PATIENCE,
    min_mut: float = CFG.GA_MIN_MUT,
    max_mut: float = CFG.GA_MAX_MUT,
    verbose: bool = True,
):
    n_features = X.shape[1]
    pop = ga_initialize(pop_size, n_features, init_prob=0.30)
    best_fit, best_ind, best_gen = -np.inf, None, 0
    hist = []
    t0 = time.time()

    for gen in range(generations):
        fits = np.array([ga_fitness(ind, X, y, model, alpha=alpha) for ind in pop])
        gbest_idx = int(np.argmax(fits))
        gbest_fit = float(fits[gbest_idx])
        gbest_ind = pop[gbest_idx].copy()
        avg_fit = float(np.mean(fits))

        if gbest_fit > best_fit:
            best_fit, best_ind, best_gen = gbest_fit, gbest_ind.copy(), gen
        elif (gen - best_gen) >= patience:
            if verbose:
                print(f"\nEarly stopping at gen {gen+1} (no improvement for {patience} gens).")
            break

        if gen > 0 and (len(hist) > 0 and gbest_fit <= hist[-1]):
            mutation_prob = min(mutation_prob * 1.25, max_mut)
        else:
            mutation_prob = max(mutation_prob * 0.95, min_mut)

        hist.append(gbest_fit)

        new_pop = []
        if elitism:
            new_pop.append(gbest_ind)
        while len(new_pop) < pop_size:
            p1 = ga_tournament_select(pop, fits, k=tournament_k)
            p2 = ga_tournament_select(pop, fits, k=tournament_k)
            c1, c2 = ga_uniform_crossover(p1, p2, prob=crossover_prob)
            c1 = ga_mutate(c1, mutation_prob)
            c2 = ga_mutate(c2, mutation_prob)
            new_pop.extend([c1, c2])
        pop = np.array(new_pop[:pop_size])

    elapsed = time.time() - t0
    return best_ind, hist, elapsed