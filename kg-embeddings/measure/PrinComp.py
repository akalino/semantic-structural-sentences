import torch
import numpy as np
import matplotlib.pyplot as plt


class PCA:
    def __init__(self, data):
        self.Data = data

    def __repr__(self):
        return f'PCA({self.data})'

    @staticmethod
    def center(_t):
        no_rows, no_columns = _t.shape
        row_means = _t.mean(axis=0)
        #Expand the matrix in order to have the same shape as X and substract, to center
        #for_subtraction = row_means.expand(no_rows, no_columns)
        X = _t - row_means #centered
        return X

    @classmethod
    def decomposition(cls, data, k):
        # Center the Data using the static method within the class
        X = cls.center(data)
        U,S,V = torch.svd(X, some=True)
        eigvecs = U.t()[:, :k]  # the first k vectors will be kept
        cls.transformed = torch.mm(X, V)
        y = torch.mm(U, eigvecs)
        # Save variables to the class object, the eigenpair and the centered data
        cls.eigenpair = (eigvecs, S)
        cls.data = X
        return y

    @staticmethod
    def explained_variance():
        # Total sum of eigenvalues (total variance explained)
        tot = sum(PCA.eigenpair[1])
        # Variance explained by each principal component
        var_exp = [(i / tot).detach().numpy() for i in sorted(PCA.eigenpair[1], reverse=True)]
        cum_var_exp = np.cumsum(var_exp)
        # X is the centered data
        X = PCA.data
        #Plot both the individual variance explained and the cumulative:
        #plt.bar(range(X.size()[1]), var_exp, alpha=0.5, align='center', label='individual explained variance')
        #plt.step(range(X.size()[1]), cum_var_exp, where='mid', label='cumulative explained variance')
        #plt.ylabel('Explained variance ratio')
        #plt.xlabel('Principal components')
        #plt.legend(loc='best')
        #plt.show()
        return var_exp
