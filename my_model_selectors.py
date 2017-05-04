import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self,
                 all_word_sequences: dict,
                 all_word_Xlengths: dict,
                 this_word: str,
                 n_constant=3,
                 min_n_components=2,
                 max_n_components=10,
                 random_state=14,
                 verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        hmm_model = None
        bic_score = None
        
        for n_components in range(self.min_n_components,
                                  self.max_n_components + 1):
            if self.verbose:
                print("running model {} with {} components".format(
                    self.this_word, n_components))
            try:
                model = GaussianHMM(n_components=n_components,
                                    covariance_type="diag",
                                    n_iter=1000,
                                    random_state=self.random_state,
                                    verbose=False).fit(self.X, self.lengths)
                logL = model.score(self.X, self.lengths)

                p = n_components ** 2 + 2 * model.n_features * n_components - 1
                
                model_bic = -2 * logL + p * math.log(len(self.X))

                if self.verbose:
                    print("BIC score for model {} is {}".format(
                        self.this_word, model_bic))
                
                if bic_score == None:
                    bic_score = model_bic
                    hmm_model = model
                elif model_bic < bic_score:
                    bic_score = model_bic
                    hmm_model = model
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(
                        self.this_word, n_components))
        return hmm_model


class SelectorDIC(ModelSelector):
    '''
    select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        hmm_model = None
        dic_score = None
        
        for n_components in range(self.min_n_components,
                                  self.max_n_components + 1):
            sumLogL = 0  # to store the logL for each other word
            
            try:
                model = GaussianHMM(n_components=n_components,
                                    covariance_type="diag",
                                    n_iter=1000,
                                    random_state=self.random_state,
                                    verbose=False).fit(self.X, self.lengths)
                logL = model.score(self.X, self.lengths)
                
                for word in self.hwords:  # loop through all but this word
                    if not word == self.this_word:
                        X, lengths = self.hwords[word]
                        sumLogL += model.score(X, lengths)
                
                model_dic = logL - ((1 / (len(self.hwords) - 1)) * sumLogL)

                if dic_score == None:
                    dic_score = model_dic
                    hmm_model = model
                elif model_dic > dic_score:
                    dic_score = model_dic
                    hmm_model = model
            except Exception as e:
                if self.verbose:
                    print("failure on {} with {} states: {}".format(
                        self.this_word, n_components, e))
        return hmm_model


class SelectorCV(ModelSelector):
    '''
    select best model based on average log Likelihood of cross-validation folds
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        hmm_model = None
        hmm_score = None
        hmm_n_component = None
        
        word_sequences = self.sequences

        try:
            # using `KFold()` with help from:
            # https://discussions.udacity.com/t/fish-word-with-selectorcv-problem/233475/3?u=brennon_york
            split_method = KFold(n_splits=min(3, len(self.lengths)))
            
            for cv_train_idx, cv_test_idx in split_method.split(word_sequences):
                if self.verbose:
                    print("Train fold indices:{} Test fold indices:{}".format(
                        cv_train_idx, cv_test_idx))  # view indices of the folds

                for n_components in range(self.min_n_components,
                                          self.max_n_components + 1):
                    try:
                        train_X, train_lengths = combine_sequences(cv_train_idx,
                                                                   word_sequences)
                        test_X, test_lengths = combine_sequences(cv_test_idx,
                                                                 word_sequences)
                    
                        model = GaussianHMM(n_components=n_components,
                                            covariance_type="diag",
                                            n_iter=1000,
                                            random_state=self.random_state,
                                            verbose=False).fit(
                                                train_X, train_lengths)

                        if self.verbose:
                            print("model created for {} with {} states".format(
                                self.this_word, n_components))

                        logL = model.score(test_X, test_lengths)

                        if self.verbose:
                            print("model logL score for {} is {}".format(self.this_word, logL))

                        if hmm_score == None:
                            hmm_score = logL
                            hmm_model = model
                        elif logL > hmm_score:
                            hmm_score = logL
                            hmm_model = model
                    except:
                        if self.verbose:
                            print("failure on {} with {} states".format(
                                self.this_word, n_components))
        except:
            return None
        return hmm_model
