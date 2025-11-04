from typing import Optional


class CFG:
    TEST_SIZE: float = 0.3
    RANDOM_STATE: int = 42
    STRATIFY: bool = True


    TOP_K_MODE: str = "auto" # "auto" or "fixed"
    TOP_K: int = 30
    K_RATIO: float = 0.5


    GA_POP: int = 24
    GA_GENS: int = 40
    GA_CROSSOVER: float = 0.9
    GA_MUTATION: float = 0.08
    GA_TOURNAMENT_K: int = 4
    GA_ELITISM: bool = True
    GA_CV: int = 3
    GA_ALPHA: float = 0.01
    GA_PATIENCE: int = 10
    GA_MIN_MUT: float = 0.01
    GA_MAX_MUT: float = 0.5


    RF_N_EST: int = 150
    RF_MAX_DEPTH: Optional[int] = None
    RF_CLASS_WEIGHT: Optional[str] = None