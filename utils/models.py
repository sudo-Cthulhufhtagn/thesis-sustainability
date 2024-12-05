from sklearn.cross_decomposition import PLSRegression
import numpy as np

class PLSPreProc(PLSRegression):
    ''' Wrapper to allow PLSRegression to be used in the Pipeline Module '''
    def __init__(self, n_components=2, scale=False, max_iter=1000, tol=1e-6, copy=True):
        super().__init__(n_components=n_components, scale=scale, max_iter=max_iter, tol=tol, copy=copy)

    def fit(self, X, y):
        # Fit the PLS model and calculate feature importances
        model = super().fit(X, y)
        self.feature_importances_ = self.vip()
        return model

    def transform(self, X):
        # Transform the data using the fitted PLS model
        return super().transform(X)

    def fit_transform(self, X, y):
        # Fit the model and transform the data in one step
        return self.fit(X, y).transform(X)
      
    def vip(model):
        # Calculate Variable Importance in Projection (VIP) scores
        t = model.x_scores_
        w = model.x_weights_
        q = model.y_loadings_
        p, h = w.shape
        vips = np.zeros((p,))
        s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        total_s = np.sum(s)
        for i in range(p):
            weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
            vips[i] = np.sqrt(p*(s.T @ weight)/total_s)
        return vips