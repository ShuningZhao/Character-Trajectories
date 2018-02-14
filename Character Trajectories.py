# Character Trajectories using different classifiers
# Shuning Zhao, z3332916
# Latest Updates: Oct 23, 2017

# packages
from __future__ import division
from __future__ import print_function
import numpy as np
from scipy.io import loadmat
import warnings
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import shuffle
from sklearn import linear_model
from sklearn import metrics, cross_validation
import scipy.io
from hmmlearn.hmm import GaussianHMM, GMMHMM

###############################################
#### PART 1: Gaussian Hidden Markov Models ####
###############################################
# ignore some warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# for regenerate the results (random states)
seed = 12
rng = np.random.RandomState(seed)
n_mix = 1

# laod data
data = loadmat('trajectories_train.mat', squeeze_me=True)
raw_X = [seq.T for seq in data['xtrain']]
y = [seq.T for seq in data['ytrain']]

# # data for GMMHMM
# # not useful, because the test results for GMMHMM are extremely unstable.
# seqs_train, y_train = shuffle(raw_X, y, random_state=rng)


# Cross validation training for the GHMMs
def GHMMs_cross_validation(X, y, N_folds=2):
    Error_rates_train = []
    Error_rates_valid = []
    MNLPs_train = []
    MNLPs_valid = []
    for i in range(N_folds):
        X_data, y_data = shuffle(X, y, random_state=rng)
        X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, test_size=0.3, random_state=rng)
        X_trains = split_20_sets(X_train, y_train)
        GHMMs = train_GHMMs(X_trains)
        y_pred_train, y_logs_train = do_prediction(GHMMs, X_train)  # training data prediction
        y_pred_valid, y_logs_valid = do_prediction(GHMMs, X_valid)  # validation data prediction
        Error_train = ER(y_pred_train, y_train)
        Error_valid = ER(y_pred_valid, y_valid)
        MNLP_train = MNLP(y_logs_train, y_train)
        MNLP_valid = MNLP(y_logs_valid, y_valid)
        Error_rates_train.append(Error_train)
        Error_rates_valid.append(Error_valid)
        MNLPs_train.append(MNLP_train)
        MNLPs_valid.append(MNLP_valid)
    Average_ER_train = np.mean(Error_rates_train)
    Average_ER_valid = np.mean(Error_rates_valid)
    Average_MNLP_train = np.mean(MNLPs_train)
    Average_MNLP_valid = np.mean(MNLPs_valid)
    return Average_ER_train, Average_ER_valid, Average_MNLP_train, Average_MNLP_valid


def split_20_sets(X_train, y_train):
    # split data into 20 different sets
    seqs = [[] for _ in range(21)]
    for i in range(len(X_train)):
        label = y_train[i]
        seqs[label].append(X_train[i])
    # vstack
    X_trains = [[] for _ in range(21)]
    for i in range(1, len(seqs)):
        X_trains[i] = np.vstack(seqs[i])

    return X_trains


def train_GHMMs(X_trains):
    # first HMM for NUMBER 1
    HMM = hmm.fit(X_trains[1])
    # fit other 19 GHMMs
    GHMMs = [[] for _ in range(21)]
    GHMMs[1] = HMM
    for i in range(2, 21):
        model = clone(GHMMs[i - 1], safe=True).fit(X_trains[i])
        GHMMs[i] = model
    # fit the first GHMM again
    GHMMs[1] = clone(GHMMs[20], safe=True).fit(X_trains[1])
    return GHMMs

# log likelihood
def log_likelihood(hmm, sequence):

    logprob_frame = hmm._compute_log_likelihood(sequence)
    logprob_sequence, _ =  hmm._do_forward_pass(logprob_frame)

    return logprob_sequence

# choose the max. likelihood
def pred(log_likelihoods):
    max_likelihood = max(log_likelihoods)
    index = log_likelihoods.index(max_likelihood)
    return index + 1

def ER(pred_list, true_list):
    count = 0
    total = len(pred_list)

    if len(pred_list) != len(true_list):
        print('cannot calculate ER because lengths of two lists are not equal')

    for i in range(len(pred_list)):
        if pred_list[i] != true_list[i]:
            count += 1
        else:
            pass
    return (count / total)


def MNLP(logs, true_list):
    MNLP = 0
    N = len(true_list)
    if len(logs) != len(true_list):
        print('MNLP not working because lengths not equal')
    for i in range(N):
        label = true_list[i] - 1
        log = logs[i][label]
        MNLP += log
    MNLP = MNLP / N
    return -MNLP

# prediction for GHMMs
def do_prediction(GHMMs, X):
    y = []
    logs_return = []
    for j in range(len(X)):
        data = X[j]
        logs = []
        for i in range (1,21):
            log = log_likelihood(GHMMs[i], data)
            logs.append(log)
        logs_return.append(logs)
        y.append(pred(logs))
    return y, logs_return



###############################################
######### PART 2: Logistic Regression #########
###############################################
# Load data
train_file = scipy.io.loadmat('trajectories_train.mat', squeeze_me=True)
test_file = scipy.io.loadmat('trajectories_xtest.mat', squeeze_me=True)
#Read the training and test data
xtrain = train_file['xtrain']
ytrain = train_file['ytrain']
xtest = test_file['xtest']
# Transpose the data
xtrain_T = [seq.T for seq in xtrain]
xtest_T = [seq.T for seq in xtest]
# Find max length for data set normalization
max_len = 0
for i in range(len(xtrain_T)):
    if max_len < len(xtrain_T[i]):
        max_len = len(xtrain_T[i])
# 0-pad xtrain and xtest so they all have the same length
i = 0
xtrain_padded = []
while i < len(xtrain):
    attr_1 = np.hstack((xtrain[i][0], [0] * (max_len - len(xtrain[i][0]))))
    attr_2 = np.hstack((xtrain[i][1], [0] * (max_len - len(xtrain[i][1]))))
    attr_3 = np.hstack((xtrain[i][2], [0] * (max_len - len(xtrain[i][2]))))
    xtrain_padded.append([attr_1, attr_2, attr_3])
    i = i + 1

i = 0
xtest_padded = []
while i < len(xtest):
    attr_1 = np.hstack((xtest[i][0], [0] * (max_len - len(xtest[i][0]))))
    attr_2 = np.hstack((xtest[i][1], [0] * (max_len - len(xtest[i][1]))))
    attr_3 = np.hstack((xtest[i][2], [0] * (max_len - len(xtest[i][2]))))
    xtest_padded.append([attr_1, attr_2, attr_3])
    i = i + 1
# Training and validation data.
i = 0
x_data = np.zeros((len(xtrain_padded),len(xtrain_padded[0])*len(xtrain_padded[0][0])))
while i < len(xtrain_padded):
    temp = np.hstack((xtrain_padded[i][0],xtrain_padded[i][1]))
    temp = np.hstack((temp,xtrain_padded[i][2] ))
    x_data[i] = temp
    i = i + 1
# New test.
i = 0
x_test = np.zeros((len(xtest_padded),len(xtest_padded[0])*len(xtest_padded[0][0])))
while i < len(xtest_padded):
    temp1 = np.hstack((xtest_padded[i][0],xtest_padded[i][1]))
    temp1 = np.hstack((temp1,xtest_padded[i][2] ))
    x_test[i] = temp1
    i = i + 1
# Making the training and Validation set for the model
x_train, x_validate, y_train, y_validate = train_test_split(x_data, ytrain, test_size=0.3, random_state=2)
# Initialize model.
print("Start training Logistic Regression model...")
logreg = linear_model.LogisticRegression()
logreg.fit(x_train, y_train)
print("Training Done.")

pred_train = logreg.predict(x_train)
train_logproba = logreg.predict_log_proba(x_train)

pred_validate = logreg.predict(x_validate)
validate_logproba = logreg.predict_log_proba(x_validate)

# MNLP for LR training
sum = 0
for i in range(0, len(y_train)):
    sum = sum + train_logproba[i][y_train[i]-1]
train_MNLP = -sum/len(y_train)

# MNLP for LR validation
sum = 0
for i in range(0, len(y_validate)):
    sum = sum + validate_logproba[i][y_validate[i]-1]
validate_MNLP = -sum/len(y_validate)

if __name__ == '__main__':
    # TEST RESULTS

    # Model 1: LR
    ####################################################################
    pred_test = logreg.predict(x_test)
    test_logproba = logreg.predict_log_proba(x_test)
    np.savetxt('predictions_LR', test_logproba, delimiter=',')
    print("The result has been saved as \"predictions_LR.txt\"")

    # Model 2: GHMMs
    #####################################################################
    # # UNCOMMNET this area to regenerate the results of
    # # cross validation results for GMMs but it will take a long time (2hrs +)
    # # n_components changes from 1 to 24, the results showed in the paper
    # for i in range(1, 25):
    #     # HMM initialization #
    #     n_states = i
    #     pi0 = np.eye(1, n_states)[0]
    #     # guess for EM
    #     trans0 = np.diag(np.ones(n_states)) + np.diag(np.ones(n_states - 1), 1)
    #     trans0 /= trans0.sum(axis=1).reshape(-1, 1)
    #     hmm = GaussianHMM(n_components=n_states,
    #                       init_params='mc',
    #                       n_iter=10,
    #                       random_state=seed)
    #     hmm.startprob_ = pi0
    #     hmm.transmat_ = trans0
    #
    #     #start training GHMMs
    #     print('Training GHMMs with', i, 'components:')
    #     Average_ER_train, Average_ER_valid, Average_MNLP_train, Average_MNLP_valid = GHMMs_cross_validation(raw_X, y)
    #     print('Training Error: ', Average_ER_train, '   ', 'Training MNLP: ', Average_MNLP_train)
    #     print('Validation Error: ', Average_ER_valid, '   ', 'Validation MNLP: ', Average_MNLP_valid)
    #####################################################################
