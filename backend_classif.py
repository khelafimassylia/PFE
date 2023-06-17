import numpy as np
import glob
import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.io
import scipy
from scipy import signal
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.manifold import Isomap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.mixture import GaussianMixture as GMM
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # for LDA classification 
from sklearn.linear_model import RidgeClassifier  # for ELM classification (view article page 4) first code ta3 ELM lta7t
import mne

import numpy as np
from scipy.stats import skew
import entropy as ent


def wilson_amplitude(signal, th):
    max_amplitude = np.max(np.abs(signal))
    th = 0.5 * max_amplitude

    x = abs(np.diff(signal))
    umbral = x >= th
    return np.sum(umbral)


def myope(signal, opts=None):
    # threshold

    max_amplitude = np.max(np.abs(signal))
    thres = 0.5 * max_amplitude

    if opts is not None and 'thres' in opts:
        thres = opts['thres']

    N = len(signal)
    Y = 0
    for i in range(N):
        if abs(signal[i]) >= thres:
            Y += 1

    MYOP = Y / N
    return MYOP


def myopulse(signal, th):
    umbral = signal >= th
    return np.sum(umbral) / len(signal)

def features_estimation(signal, fs, frame, step):
    """
    Compute time, frequency and time-frequency features from signal.
    :param signal: numpy array signal.
    :param channel_name: string variable with the EMG channel name in analysis.
    :param fs: int variable with the sampling frequency used to acquire the signal
    :param frame: sliding window size
    :param step: sliding window step size
    :param plot: boolean variable to plot estimated features.
    :return: total_feature_matrix -- python Data-frame with.
    :return: features_names -- python list with
    """

    features_names = ['SSC', 'IEMG', 'MYOP','SKW', 'RMS', 'MAV', 'WL', 'ZC', 'ACT', 'COMP', 'MOB', 'FR',
                       'MNP', 'TOT', 'MNF','PKF']
    #,ssc, iemg, wamp, acc, dasdv, myop, skw, rms, mav, wl, zc, activity, complexity, mobility
    # fr,mnp,tot,mnf,mdf,pkf

    time_matrix = features_artcile_implementation(signal, frame, step)
    frequency_matrix = frequency_features_estimation(signal, fs, frame, step)
    #, frequency_matrix
    total_feature_matrix = pd.DataFrame(np.column_stack((time_matrix, frequency_matrix)),
                                        columns=features_names)

    #print('EMG features were from channel {} extracted successfully'.format(channel_name))

    #if plot:
    #    plot_features(signal, channel_name, fs, total_feature_matrix, step)

    return total_feature_matrix

def features_artcile_implementation(signal, frame, step):
    """
    Compute different features from signal using sliding window method.
    :param signal: numpy array signal.
    :param frame: sliding window size.
    :param step: sliding window increment.
    :return: time_features_matrix: narray matrix with the time features stacked by columns.
    """

    #variance = []  # good
    iemg = []  # good
    #wamp = []  # good
    zc = []  # zero crossing                  #good
    wl = []  # wavelenght                     #good
    ssc = []  # slope sign change             #good
    skw = []  # skewness                      #good
    rms = []  # root mean square              #good
    mav = []  # mean absolute value           #good
    #iav = []  # integral absolute value       #good
    complexity = []  # good
    mobility = []  # good
    activity = []  # good
    #acc = []
    #dasdv = []
    myop = []  # good

    # max_amplitude = np.max(np.abs(signal))
    # th = 0.5 * max_amplitude
    th=0

    for i in range(frame, signal.size+step, step):

        x = signal[i - frame:i]

        #variance.append(np.var(x))
        iemg.append(np.sum(abs(x)))  # Integral
        #wamp.append(wilson_amplitude(x, th))  # Willison amplitude
        ssc.append(slope_sign_change(x))
        skw.append(skew(x))
        #acc.append(np.mean(np.sum(np.abs(np.diff(x)))))
        #dasdv.append(np.mean(np.sum(np.abs(np.diff(x)))))
        rms.append(np.sqrt(np.mean(x ** 2)))
        #iav.append(np.sum(abs(x)))  # Integral
        mav.append(np.sum(np.absolute(x)) / frame)  # Mean Absolute Value
        wl.append(np.sum(abs(np.diff(x))))  # Wavelength
        zc.append(zcruce(x, th))  # Zero-Crossing
        myop.append(myope(x))

        comp, mob, act = hjorth_param(x)

        complexity.append(comp)
        mobility.append(mob)
        activity.append(act)

        #features_names = ['SSC', 'SKW', 'RMS', 'IAV', 'MAV', 'WL', 'ZC', "COMP", "MOB", "ACT"]
        #features_names= ['VAR', 'IEMG', 'WAMP', 'MYOP', 'ZC', 'SSC', 'RMS', 'IAV', 'MAV', 'WL', 'SKW', "COMP", "MOB", "ACT"]
        #features_names= ['IEMG', 'MYOP', 'IAV',  'WL']
        #dtype = np.dtype([('var', float), ('iemg', float), ('wamp', float), ('myop', float), ('zc', float), ('ssc', float), ('rms', float), ('iav', float), ('mav', float), ('wl', float), ('skw', float), ('cmplx', float), ('mobil', float), ('activ', float)])
    time_features_matrix = np.column_stack(
        (ssc, iemg, myop, skw, rms, mav, wl, zc, activity, complexity, mobility)) # a total of 13 features
    return time_features_matrix
# compute the zero crossing
#  number of zero crossings (1feature), waveform length (1 feature), number of slopesign changes (1 feature), skewness (1 feature), root-meansquare (1 feature), mean absolute value (1 feature), integral absolute value (1 feature), parameters of an autoregressive (AR) model with an order of 11 providing significant enhancements upon smaller model orders (11 features), and the Hjorth time-domain parameters (3 features)

def frequency_features_estimation(signal, fs, frame, step):
    """
    Compute frequency features from signal using sliding window method.
    :param signal: numpy array signal.
    :param fs: sampling frequency of the signal.
    :param frame: sliding window size
    :param step: sliding window step size
    :return: frequency_features_matrix: narray matrix with the frequency features stacked by columns.
    """

    fr = []
    mnp = []  # not good
    tot = []  # not good
    mnf = []  # not good
    #mdf = []  # not good
    pkf = []  # not good

    for i in range(frame, signal.size+step, step):
        x = signal[i - frame:i]
        frequency, power = spectrum(x, fs)

        fr.append(frequency_ratio(frequency, power))  # Frequency ratio
        mnp.append(np.sum(power) / len(power))  # Mean power
        tot.append(np.sum(power))  # Total power
        mnf.append(mean_freq(frequency, power))  # Mean frequency
        #mdf.append(median_freq(frequency, power))  # Median frequency
        pkf.append(frequency[power.argmax()])  # Peak frequency

    frequency_features_matrix = np.column_stack((fr,mnp,tot,mnf,pkf))

    return frequency_features_matrix

def spectrum(signal, fs):
    m = len(signal)
    n = next_power_of_2(m)
    y = np.fft.fft(signal, n)
    yh = y[:n//2]
    dt = 1/fs
    freq = np.fft.fftfreq(n, dt)[:n//2]
    power = np.abs(yh)**2 / (n**2)
    return freq, power

def frequency_ratio(frequency, power):
    power_low = power[(frequency >= 30) & (frequency <= 250)]
    power_high = power[(frequency > 250) & (frequency <= 500)]
    ULC = np.sum(power_low)
    UHC = np.sum(power_high)

    return ULC / UHC

def mean_freq(frequency, power):
    num = 0
    den = 0
    for i in range(int(len(power) / 2)):
        num += frequency[i] * power[i]
        den += power[i]

    return num / den

def median_freq(frequency, power):
    power_total = np.sum(power) / 2
    temp = 0
    tol = 0.01
    errel = 1
    i = 0

    while abs(errel) > tol:
        temp += power[i]
        errel = (power_total - temp) / power_total
        i += 1
        if errel < 0:
            errel = 0
            i -= 1

    return frequency[i]

def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def zcruce(X, th):

    cruce = 0
    for cont in range(len(X) - 1):
        can = X[cont] * X[cont + 1]
        can2 = abs(X[cont] - X[cont + 1])
        if can < 0 and can2 > th:
            cruce = cruce + 1
    return cruce


def slope_sign_change(signal):
    """
    Computes the slope sign change of a signal.

    Args:
        signal (numpy.ndarray): 1D array of signal values.

    Returns:
        int: The number of times the slope of the signal changes sign.
    """
    num_sign_changes = 0
    for i in range(1, len(signal) - 1):
        if (signal[i] > signal[i-1] and signal[i] > signal[i+1]) or \
           (signal[i] < signal[i-1] and signal[i] < signal[i+1]):
            num_sign_changes += 1
    return num_sign_changes
# rms
# parameters of an autoregressive (AR) model

# Hjorth time-domain parameters


def hjorth_param(signal):
    # calculate activity
    act = np.var(signal)

    # calculate mobility                 #good
    diff_signal = np.diff(signal)
    var_diff_signal = np.var(diff_signal)
    mob = np.sqrt(var_diff_signal / np.var(signal))

    # calculate complexity                      #good
    diff_diff_signal = np.diff(diff_signal)
    var_diff_diff_signal = np.var(diff_diff_signal)
    comp = np.sqrt(var_diff_diff_signal / var_diff_signal) / \
        mob if mob > 0 else 0

    return comp, mob, act

def load_all_electrodes(path):
    #colnames=['electrode1', 'electrode2', 'electrode3', 'electrode4','electrode5','electrode6','electrode8','electrode9']
    #path = 'C:/Users/massy/OneDrive/Bureau/THE REAL PFE/DATASET/S1-Delsys-15Class'
    all_files = glob.glob(os.path.join(path, "*.csv"))
    labels = []
    dfs = []
    for file in all_files:
        df = pd.read_csv(file, header=None)
        arr = df.transpose().values.ravel()  # flatten the DataFrame into a 1D array
        # reshape the 1D array into a 1x640000 DataFrame
        result = pd.DataFrame(arr.reshape(1, -1))
        filename = os.path.basename(file)
        label = os.path.splitext(filename)[0][0:2]
        labels.append(label)
        #df['label'] = os.path.splitext(filename)[0]
        dfs.append(result)

    data = pd.concat(dfs, ignore_index=1)
    return data, labels




def classif (folder):
    start_time=time.time()
    data_all,label_all=load_all_electrodes(folder)

    b=data_all.replace(np.nan,500)


    bpdata = mne.filter.filter_data(b.to_numpy(), sfreq=1000, l_freq=20, h_freq=450)
    signalfiltred = mne.filter.notch_filter(bpdata, 1000, 50)
    signalfiltred=pd.DataFrame(signalfiltred)


    samples=5000
    dff=[]
    for electrodes in range(0,len(signalfiltred.iloc[0]),5000):
        #print(electrodes)
        electrode=signalfiltred.iloc[:, electrodes:electrodes+samples]
        df=pd.concat(electrode.apply(lambda row: features_estimation(row.values, 1000, 128, 64), axis=1).tolist())
        #df=pd.concat(electrode.apply(lambda row: pd.DataFrame(features_artcile_implementation(row.values, 128, 64),columns=features_names), axis=1).tolist())
        #df=pd.concat(electrode.apply(lambda row: pd.DataFrame(feature_segmentation(row.values, 50, 25,feature),columns=feature), axis=1).tolist())
        dff.append(df)
    final=pd.concat(dff,axis=1)


    #ADD LABELS
    import itertools
    number_of_electrodes=2
    window_size=128
    increment=64
    nbr_trial=10
    number_of_new_samples=(((len(signalfiltred.iloc[0])//number_of_electrodes)- window_size)//increment)+2
    l=np.unique(label_all)
    new_labels = np.repeat(l,number_of_new_samples*nbr_trial)

    scaled_df = pd.DataFrame(final, columns=final.columns)
    y=new_labels

    X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, test_size=0.2, random_state=42)

    #X_train, X_val_test, y_train, y_val_test = train_test_split(scaled_df, y, test_size=0.2, random_state=42)
    #X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

    y_test=np.ravel(y_test)
    y_train=np.ravel(y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=6)
    knn.fit(X_train, y_train)
    yKNN_tr=knn.predict(X_train)
    yKNN_pred = knn.predict(X_test)


    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    yRF_tr=rf.predict(X_train)
    yRF_pred = rf.predict(X_test)


    clf = SVC(kernel='rbf',gamma='auto',C=10)
    clf.fit(X_train, y_train)
    ySVM_tr=clf.predict(X_train)
    ySVM_pred = clf.predict(X_test)



    lda= LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    yLDA_tr=lda.predict(X_train)
    yLDA_pred = lda.predict(X_test)


    logistic_reg = LogisticRegression(max_iter=1000)
    logistic_reg.fit(X_train, y_train)
    yLOG_tr=logistic_reg.predict(X_train)
    yLOG_pred = logistic_reg.predict(X_test)
    end_time=time.time()
    elapsed_time=end_time-start_time
    classifier_clf = [knn, rf, clf, lda, logistic_reg]
    classifiers=['KNN',"Random forest","SVM","LDA","Logistic regression"]
    accuracy_test=[accuracy_score(y_test,yKNN_pred),accuracy_score(y_test,yRF_pred),accuracy_score(y_test,ySVM_pred),accuracy_score(y_test,yLDA_pred),accuracy_score(y_test,yLOG_pred)]
    accuracy_train=[accuracy_score(y_train,yKNN_tr),accuracy_score(y_train,yRF_tr),accuracy_score(y_train,ySVM_tr),accuracy_score(y_train,yLDA_tr),accuracy_score(y_train,yLOG_tr)]
    print(elapsed_time)
    return classifiers,accuracy_train,accuracy_test, classifier_clf

def select_classifier(classifier,clf):
    import joblib
    #saving le model (BEST ONE)
    filename = f'{classifier}.sav'
    joblib.dump(clf, filename)
    return 1

