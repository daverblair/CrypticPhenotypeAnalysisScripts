import pandas as pd
import statsmodels.api as sm
import numpy as np
import copy
from scipy import stats
from scipy import special

def logsig(x):
    """Compute the log-sigmoid function component-wise."""
    out = np.zeros_like(x)
    idx0 = x < -33
    out[idx0] = x[idx0]
    idx1 = (x >= -33) & (x < -18)
    out[idx1] = x[idx1] - np.exp(x[idx1])
    idx2 = (x >= -18) & (x < 37)
    out[idx2] = -np.log1p(np.exp(-x[idx2]))
    idx3 = x >= 37
    out[idx3] = -np.exp(-x[idx3])
    return out



class FirthRegression:

    def firth_likelihood(self,beta):
        """
        Note: likelihood based on methods in http://fa.bianp.net/blog/2019/evaluate_logistic/
        This avoids numerical precision issues.
        """
        Z=np.dot(self.model.exog,beta)
        return (-1.0*np.sum((1 - self.model.endog) * Z - logsig(Z)))+0.5*np.linalg.slogdet(-1.0*self.model.hessian(beta))[1]

    def __init__(self,data_table,x_variables,y_variable,hasconst=False):
        """

        Basic implementation of Firth-penalized logistic regression. Based on the implementation from John Lees: https://gist.github.com/johnlees/3e06380965f367e4894ea20fbae2b90d and the methods described in PMID: 12758140. Note, this implementation is not optimized for speed. Many improvements could be made.

        Parameters
        ----------
        data_table : pd.DataFrame
            Pandas data frame containing endogenous and exogenous variables.
        x_variables : list
            Exogenous variables to use in the regression. Expects list of strings.
        y_variable : string
            Engogenous varabile in data_table
        hasconst : bool
            Indicates whether dataframe contains intercept/constant. Default is False.

        Returns
        -------
        FirthRegression class

        """

        self.hasconst=hasconst
        self.x_variables=x_variables

        self.X=data_table[x_variables].values
        if self.hasconst==False:
            self.X=np.hstack((np.ones((self.X.shape[0],1)),self.X))
            self.x_variables=['Intercept']+self.x_variables
            self.const_column=0
        else:
            self.const_column=None
            for i in range(self.X.shape[1]):
                if len(np.setdiff1d(self.X[:,i],np.array([1])))==0:
                    self.const_column=i
            if self.const_column is None:
                raise ValueError("Must include constant in data table if hasconst=True")

        self.Y=data_table[y_variable].values.reshape(-1,1)
        self.model=sm.Logit(self.Y, self.X)
        self.firth_model=None

    def FirthInference(self,variables,start_vec=None, num_iters=1000,step_limit=100, convergence_limit=1e-6):
        """
        Performs Logistic regerssion infernece using penalized likelihood ratio test for a single/multiple variables. Note, can be called repeatedly for different variables. Full model will only be fit once.

        Parameters
        ----------
        variables : list of strings or a single string
            Variables for inference. Can include only single string, which will be transformed into a list
        start_vec : np.array
            Initial vector for parameters. Be careful, poor initialization can result in optimization failure.
        num_iters: int
            Number of Newton-Rapheson iterations
        step_limit : int
            Number of steps to allow for step-halving. Default is 100.
        convergence_limit : float
            Threshold for convergence. Based on the norm of the difference between the new and old paramater vector

        Returns
        -------
        Dict
            Model Log-likelihood
            Table of parameters and their associated effect coefficiencts and standard errors
            P-value for model with free vs restricted (BETA=0) parameters; likelihood ratio test

        """
        if self.firth_model is None:
            self.firth_model = self.fitModel(start_vec, step_limit,num_iters, convergence_limit)

        if isinstance(variables,list)==False:
            variables=[variables]

        test_variable_indices=[self.x_variables.index(variable) for variable in variables]
        null_blocking_vec=np.ones(self.X.shape[1],dtype=np.float64)

        for test_variable in test_variable_indices:
            null_blocking_vec[test_variable]=0.0
        null_model=self.fitModel(start_vec, step_limit,num_iters,convergence_limit,blocking_vec=null_blocking_vec)
        p_val =stats.chi2.sf(2.0*(self.firth_model['LogLike'] - null_model['LogLike']), len(variables))

        return_model=copy.deepcopy(self.firth_model)
        return_model['PVal']=p_val

        return return_model

    def fitModel(self, start_vec, step_limit,num_iters, convergence_limit,blocking_vec=None):
        # if start_vec is None, then initialize with zeros plus intercept set to log-odds of incidence
        if start_vec is None:
            start_vec = np.zeros(self.X.shape[1])
            start_vec[self.const_column]=np.log(np.mean(self.Y)/(1.0-np.mean(self.Y)))
        else:
            assert start_vec.shape[0]==self.X.shape[1],"Shape of parameter initialization vector does not match the number of covariates. Don't forget to include the intercept."

        #if blocking_vec is None, then assume unblocked
        if blocking_vec is None:
            blocking_vec=np.ones(start_vec.shape[0],dtype=np.float64)
        else:
            assert blocking_vec.shape[0]==self.X.shape[1],"Shape of parameter blocking vector does not match the number of covariates. Don't forget to include the intercept."

        beta_iterations = []
        beta_iterations.append(start_vec)

        for i in range(0, int(num_iters)):
            #based on implementation in PMID: 12758140
            pi = self.model.predict(beta_iterations[i])
            W_diag = pi*(1-pi)
            var_covar_mat = np.linalg.pinv(-self.model.hessian(beta_iterations[i]))

            root_W_diag=np.sqrt(W_diag)
            H=np.transpose(self.X*root_W_diag.reshape(-1,1))
            H=np.matmul(var_covar_mat, H)
            H_diag = np.sum(self.X*root_W_diag.reshape(-1,1)*H.T,axis=1)

            U = np.matmul(np.transpose(self.X), self.Y - pi.reshape(-1,1) + (H_diag*(0.5 - pi)).reshape(-1,1))

            new_beta = beta_iterations[i] + np.matmul(var_covar_mat, U).T.ravel()*blocking_vec
            # step halving
            j = 0
            while self.firth_likelihood(new_beta) < self.firth_likelihood(beta_iterations[i]):

                new_beta = beta_iterations[i] + 0.5*(new_beta - beta_iterations[i])*blocking_vec
                j = j + 1
                if (j > step_limit):
                    if (i > 0):
                        print('Warning: Unable to find parameter vector to further optimize likelihood at iteration {0:d}. Convergence Uncertain.\n'.format(i))
                        new_beta=beta_iterations[i]
                    else:
                        raise ValueError("Unable to find parameter vector to optimize likelihood on first iteration. Try increasing step_limit.")
            beta_iterations.append(new_beta)
            if (np.linalg.norm(beta_iterations[-1] - beta_iterations[-2]) < convergence_limit):
                break
        if np.linalg.norm(beta_iterations[-1] - beta_iterations[-2]) >= convergence_limit:
            raise ValueError('Firth regression failed failed to converge in {0:d} iterations. Consider increasing iteration number.\n'.format(i+1))
        else:
            fitll = self.firth_likelihood(beta_iterations[-1])
            bse = np.sqrt(np.diagonal(np.linalg.pinv(-self.model.hessian(beta_iterations[-1]))))
            output={'VAR':[],'BETA':[],'SE':[]}
            for i,var in enumerate(self.x_variables):
                output['VAR']+=[var]
                output['BETA']+=[beta_iterations[-1][i]]
                output['SE']+=[bse[i]]

            output=pd.DataFrame(output)
            output.set_index('VAR',inplace=True)
            return {'LogLike':fitll,'ParamTable':output}
