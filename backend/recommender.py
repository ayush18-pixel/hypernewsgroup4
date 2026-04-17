import numpy as np

class LinUCBBandit:
    def __init__(self, n_features: int, alpha: float = 0.5):
        self.alpha = alpha
        self.A = {}   # n_features x n_features per article
        self.b = {}   # n_features vector per article

    def get_ucb_score(self, article_id: str, context: np.ndarray) -> float:
        if article_id not in self.A:
            d = len(context)
            self.A[article_id] = np.identity(d)
            self.b[article_id] = np.zeros(d)

        A_inv = np.linalg.inv(self.A[article_id])
        theta = A_inv @ self.b[article_id]
        reward_est = theta @ context
        conf_bound = self.alpha * np.sqrt(context @ A_inv @ context)
        return reward_est + conf_bound

    def update(self, article_id: str, context: np.ndarray, reward: float):
        if article_id not in self.A:
            d = len(context)
            self.A[article_id] = np.identity(d)
            self.b[article_id] = np.zeros(d)
        
        self.A[article_id] += np.outer(context, context)
        self.b[article_id] += reward * context
