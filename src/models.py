"""
Implementations of models from scratch:
- LogisticRegressionScratch
- GaussianNaiveBayes
- KNearestNeighbors
- DecisionTree (simple CART)
"""
import numpy as np
from collections import Counter, defaultdict
import pickle

# -------------------------
# Logistic Regression
# -------------------------
class LogisticRegressionScratch:
    def __init__(self, lr=0.1, n_iter=1000, fit_intercept=True, l2=0.0, verbose=False):
        self.lr = lr
        self.n_iter = n_iter
        self.fit_intercept = fit_intercept
        self.l2 = l2
        self.verbose = verbose

    def _add_intercept(self, X):
        if not self.fit_intercept:
            return X
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def fit(self, X, y):
        X = self._add_intercept(X)
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        for i in range(self.n_iter):
            z = X.dot(self.w)
            preds = 1 / (1 + np.exp(-z))
            error = preds - y
            grad = X.T.dot(error) / X.shape[0]
            if self.l2 > 0:
                grad += (self.l2 * self.w) / X.shape[0]
            self.w -= self.lr * grad
            if self.verbose and i % (self.n_iter // 10 + 1) == 0:
                loss = -np.mean(y * np.log(preds + 1e-12) + (1-y)*np.log(1-preds+1e-12))
                print(f"iter {i}, loss {loss:.4f}")

    def predict_proba(self, X):
        X = self._add_intercept(X)
        z = X.dot(self.w)
        probs = 1 / (1 + np.exp(-z))
        return probs

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path):
        np.save(path, self.w)

    def load(self, path):
        self.w = np.load(path)

# -------------------------
# Gaussian Naive Bayes
# -------------------------
class GaussianNaiveBayes:
    def __init__(self):
        self.class_priors = {}
        self.mean = {}
        self.var = {}
        self.cat_counts = {}

    def fit(self, X, y, feature_types):
        """
        feature_types: list of 'num' or 'cat' for each feature
        X: pandas-like (2D np array or list of lists)
        """
        X = np.array(X)
        self.feature_types = feature_types
        classes = np.unique(y)
        for c in classes:
            Xc = X[y==c]
            self.class_priors[c] = Xc.shape[0] / X.shape[0]
            self.mean[c] = {}
            self.var[c] = {}
            self.cat_counts[c] = {}
            for j, t in enumerate(feature_types):
                col = Xc[:, j]
                if t == 'num':
                    col = col.astype(float)
                    self.mean[c][j] = np.mean(col)
                    self.var[c][j] = np.var(col) + 1e-9
                else:
                    # categorical: store counts
                    counts = {}
                    for val in col:
                        counts[val] = counts.get(val, 0) + 1
                    self.cat_counts[c][j] = counts

    def _gaussian_prob(self, x, mean, var):
        coef = 1.0 / np.sqrt(2.0 * np.pi * var)
        exponent = - ((x - mean) ** 2) / (2 * var)
        return coef * np.exp(exponent)

    def predict_proba(self, X):
        X = np.array(X)
        probs = []
        for xi in X:
            class_scores = {}
            for c in self.class_priors:
                score = np.log(self.class_priors[c] + 1e-12)
                for j, t in enumerate(self.feature_types):
                    val = xi[j]
                    if t == 'num':
                        mean = self.mean[c][j]
                        var = self.var[c][j]
                        p = self._gaussian_prob(float(val), mean, var) + 1e-12
                        score += np.log(p)
                    else:
                        counts = self.cat_counts[c].get(j, {})
                        # Laplace smoothing
                        count_val = counts.get(val, 0) + 1
                        total = sum(counts.values()) + len(counts) + 1
                        p = count_val / total
                        score += np.log(p)
                class_scores[c] = score
            # convert log-scores to probabilities
            max_log = max(class_scores.values())
            exps = {c: np.exp(score - max_log) for c, score in class_scores.items()}
            s = sum(exps.values())
            probs.append({c: exps[c]/s for c in exps})
        # return probability for class 1
        return np.array([p.get(1, 0.0) for p in probs])

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def save(self, path):
        np.savez(path, class_priors=self.class_priors, mean=self.mean, var=self.var, cat_counts=self.cat_counts, feature_types=self.feature_types)

    def load(self, path):
        data = np.load(path, allow_pickle=True)
        self.class_priors = data['class_priors'].item()
        self.mean = data['mean'].item()
        self.var = data['var'].item()
        self.cat_counts = data['cat_counts'].item()
        self.feature_types = data['feature_types'].tolist()

# -------------------------
# K-Nearest Neighbors
# -------------------------
class KNearestNeighbors:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        preds = []
        for x in X:
            # Euclidean distances
            dists = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            idx = np.argsort(dists)[:self.k]
            votes = self.y_train[idx]
            vals, counts = np.unique(votes, return_counts=True)
            preds.append(vals[np.argmax(counts)])
        return np.array(preds)

    def save(self, path):
        np.savez(path, X=self.X_train, y=self.y_train, k=self.k)

    def load(self, path):
        d = np.load(path)
        self.X_train = d['X']
        self.y_train = d['y']
        self.k = int(d['k'])

# -------------------------
# Decision Tree (CART) - simple numeric splits only for brevity
# -------------------------
class DecisionNode:
    def __init__(self, gini=None, num_samples=None, num_samples_per_class=None, predicted_class=None):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def _gini(self, y):
        m = len(y)
        if m == 0:
            return 0
        counts = np.bincount(y)
        prob = counts / m
        return 1.0 - np.sum(prob ** 2)

    def _best_split(self, X, y):
        m, n = X.shape
        if m < 2:
            return None, None
        best_gini = 1.0
        best_idx, best_thr = None, None
        for idx in range(n):
            vals = X[:, idx]
            # consider unique thresholds
            thresholds = np.unique(vals)
            if len(thresholds) == 1:
                continue
            # try midpoints
            for j in range(len(thresholds)-1):
                thr = (thresholds[j] + thresholds[j+1]) / 2.0
                left_mask = vals <= thr
                y_left = y[left_mask]
                y_right = y[~left_mask]
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                gini_left = self._gini(y_left)
                gini_right = self._gini(y_right)
                gini = (len(y_left) * gini_left + len(y_right) * gini_right) / m
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = thr
        return best_idx, best_thr

    def _build_tree(self, X, y, depth=0):
        node = DecisionNode()
        node.num_samples = y.size
        node.num_samples_per_class = np.bincount(y, minlength=2)
        node.predicted_class = np.argmax(node.num_samples_per_class)
        node.gini = self._gini(y)

        if depth < self.max_depth and node.num_samples >= self.min_samples_split and node.gini > 0:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                node.feature_index = idx
                node.threshold = thr
                mask = X[:, idx] <= thr
                node.left = self._build_tree(X[mask], y[mask], depth+1)
                node.right = self._build_tree(X[~mask], y[~mask], depth+1)
        return node

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.tree_ = self._build_tree(np.array(X), np.array(y))

    def _predict_one(self, x, node):
        if node.left is None and node.right is None:
            return node.predicted_class
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_one(x, self.tree_) for x in X])

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.tree_, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.tree_ = pickle.load(f)
