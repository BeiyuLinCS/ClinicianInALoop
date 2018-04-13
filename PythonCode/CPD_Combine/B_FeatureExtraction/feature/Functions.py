#!/usr/bin/python

# THIS SOFTWARE IS DISTRIBUTED UNDER GPL LICENSE:
# http://www.gnu.org/licenses/gpl.txt

#from pylab import *
import numpy as np
from scipy import linalg
from scipy.stats import norm
from collections import Counter
from collections import defaultdict
from sklearn import preprocessing
import pickle
import datetime
import xlwt
from sklearn import tree, cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier


def get_date_time(str_date, str_time):
    """
    Global Function: getDataTime(str_date, str_time):
    Description:
        Take Data String and Time String, spinning out a
        datetime object corresponding to the date and time
        provided.
    :param str_date: Date String D-M-Y
    :param str_time: Time String with Format H:M:S
    :return datetime: Converting Date, Time string into a
        datetime variable
    """
    data_list = str_date.split('-')
    time_list = str_time.split(':')
    sec_list = time_list[2].split('.')
    #print("str_date", str_date)
    #print("str_time", str_time)
    if (len(sec_list) == 1):
	sec_list.append("0000")
    dt = datetime.datetime(int(data_list[0]),
                           int(data_list[1]),
                           int(data_list[2]),
                           int(time_list[0]),
                           int(time_list[1]),
                           int(sec_list[0]),
                           int(sec_list[1]))
    return dt

def compmedDist(X):
    size1 = X.shape[0];
    Xmed = X;

    G = sum((Xmed * Xmed), 1);
    Q = np.tile(G[:,np.newaxis], (1, size1));
    R = np.tile(G, (size1, 1));
    print("Xmed",  type(Xmed))
    print("Xmed.T", type(Xmed.T))
    print("Q,R,SHAPE,xmed.shape", G.shape, Q.shape, R.shape,np.dot(Xmed, Xmed.T).shape)
    dists = Q + R - 2 * np.dot(Xmed, Xmed.T);
    dists = dists - np.tril(dists);
    dists = dists.reshape(size1 ** 2, 1, order='F').copy();
    if sum(dists) > 0:
        qqq = sqrt(0.5 * median(dists[dists > 0]));
    else:
        qqq = 1
    return qqq

def kernel_Gaussian(x, c, sigma):
    (d, nx) = x.shape
    (d, nc) = c.shape
    x2 = sum(x ** 2, 0)
    c2 = sum(c ** 2, 0)

    distance2 = np.tile(c2, (nx, 1)) + \
                np.tile(x2[:, np.newaxis], (1, nc)) \
                - 2 * np.dot(x.T, c)

    return exp(-distance2 / (2 * (sigma ** 2)));

def R_ULSIF(x_nu, x_de, x_re, alpha, sigma_list, lambda_list, b, fold):
    # x_nu: samples from numerator
    # x_de: samples from denominator
    # x_re: reference sample
    # alpha: alpha defined in relative density ratio
    # sigma_list, lambda_list: parameters for model selection
    # b: number of kernel basis
    # fold: number of fold for cross validation

    (d, n_nu) = x_nu.shape;
    (d, n_de) = x_de.shape;

    b = min(b, n_nu);

    x_ce = x_nu[:, r_[0:b]]

    score_cv = np.zeros((size(sigma_list), \
                      size(lambda_list)));

    cv_index_nu = permutation(n_nu)

    cv_split_nu = floor(r_[0:n_nu] * fold / n_nu)
    cv_index_de = permutation(n_de)

    cv_split_de = floor(r_[0:n_de] * fold / n_de)

    for sigma_index in r_[0:size(sigma_list)]:
        sigma = sigma_list[sigma_index];
        K_de = kernel_Gaussian(x_de, x_ce, sigma).T;
        K_nu = kernel_Gaussian(x_nu, x_ce, sigma).T;

        score_tmp = np.zeros((fold, size(lambda_list)));

        for k in r_[0:fold]:
            Ktmp1 = K_de[:, cv_index_de[cv_split_de != k]];
            Ktmp2 = K_nu[:, cv_index_nu[cv_split_nu != k]];

            Ktmp = alpha / Ktmp2.shape[1] * np.dot(Ktmp2, Ktmp2.T) + \
                   (1 - alpha) / Ktmp1.shape[1] * np.dot(Ktmp1, Ktmp1.T);

            mKtmp = mean(K_nu[:, cv_index_nu[cv_split_nu != k]], 1);

            for lambda_index in r_[0:size(lambda_list)]:
                lbd = lambda_list[lambda_index];

                thetat_cv = linalg.solve(Ktmp + lbd * np.eye(b), mKtmp);
                thetah_cv = thetat_cv;

                score_tmp[k, lambda_index] = alpha * mean(
                    np.dot(K_nu[:, cv_index_nu[cv_split_nu == k]].T, thetah_cv) ** 2) / 2. \
                                             + (1 - alpha) * mean(
                    np.dot(K_de[:, cv_index_de[cv_split_de == k]].T, thetah_cv) ** 2) / 2. \
                                             - mean(np.dot(K_nu[:, cv_index_nu[cv_split_nu == k]].T, thetah_cv));

            score_cv[sigma_index, :] = mean(score_tmp, 0);

    score_cv_tmp = score_cv.min(1);
    lambda_chosen_index = score_cv.argmin(1);

    sigma_chosen_index = score_cv_tmp.argmin();

    lambda_chosen = lambda_list[lambda_chosen_index[sigma_chosen_index]];
    sigma_chosen = sigma_list[sigma_chosen_index];

    K_de = kernel_Gaussian(x_de, x_ce, sigma_chosen).T;
    K_nu = kernel_Gaussian(x_nu, x_ce, sigma_chosen).T;

    coe = alpha * np.dot(K_nu, K_nu.T) / n_nu + \
          (1 - alpha) * np.dot(K_de, K_de.T) / n_de + \
          lambda_chosen * np.eye(b)
    var = mean(K_nu, 1)

    thetat = linalg.solve(coe, var);

    thetah = thetat;
    wh_x_de = np.dot(K_de.T, thetah).T;
    wh_x_nu = np.dot(K_nu.T, thetah).T;

    wh_x_de[wh_x_de < 0] = 0

    PE = mean(wh_x_nu) - 1. / 2 * (alpha * mean(wh_x_nu ** 2) + \
                                   (1 - alpha) * mean(wh_x_de ** 2)) - 1. / 2;

    return (PE)

def Sep_CP(x_nu, x_de, sigma_list, lambda_list, b, fold):
    # x_nu: samples from numerator
    # x_de: samples from denominator
    # x_re: reference sample
    # alpha: alpha defined in relative density ratio
    # sigma_list, lambda_list: parameters for model selection
    # b: number of kernel basis
    # fold: number of fold for cross validation

    (d, n_nu) = x_nu.shape;
    (d, n_de) = x_de.shape;

    b = min(b, n_nu);

    x_ce = x_nu[:, r_[0:b]]

    score_cv = np.zeros((size(sigma_list), \
                      size(lambda_list)));

    cv_index_nu = permutation(n_nu)

    cv_split_nu = floor(r_[0:n_nu] * fold / n_nu)
    cv_index_de = permutation(n_de)

    cv_split_de = floor(r_[0:n_de] * fold / n_de)

    for sigma_index in r_[0:size(sigma_list)]:
        sigma = sigma_list[sigma_index];
        K_de = kernel_Gaussian(x_de, x_ce, sigma).T;
        K_nu = kernel_Gaussian(x_nu, x_ce, sigma).T;

        score_tmp = np.zeros((fold, size(lambda_list)));

        for k in r_[0:fold]:
            Ktmp1 = K_de[:, cv_index_de[cv_split_de != k]];
            Ktmp2 = K_nu[:, cv_index_nu[cv_split_nu != k]];

            Ktmp = 1 / Ktmp1.shape[1] * np.dot(Ktmp1, Ktmp1.T);

            mKtmp = mean(K_nu[:, cv_index_nu[cv_split_nu != k]], 1);

            for lambda_index in r_[0:size(lambda_list)]:
                lbd = lambda_list[lambda_index];

                thetat_cv = linalg.solve(lbd * np.eye(b), mKtmp);
                thetah_cv = thetat_cv;

                score_tmp[k, lambda_index] = abs(1 - mean(np.dot(K_nu[:, cv_index_nu[cv_split_nu == k]].T, thetah_cv)));

            score_cv[sigma_index, :] = mean(score_tmp, 0);

    score_cv_tmp = score_cv.min(1);
    lambda_chosen_index = score_cv.argmin(1);

    sigma_chosen_index = score_cv_tmp.argmin();

    lambda_chosen = lambda_list[lambda_chosen_index[sigma_chosen_index]];
    sigma_chosen = sigma_list[sigma_chosen_index];

    K_de = kernel_Gaussian(x_de, x_ce, sigma_chosen).T;
    K_nu = kernel_Gaussian(x_nu, x_ce, sigma_chosen).T;

    coe = lambda_chosen * np.eye(b)
    var = mean(K_nu, 1)

    thetat = linalg.solve(coe, var);

    thetah = thetat;
    wh_x_de = np.dot(K_de.T, thetah).T;
    wh_x_nu = np.dot(K_nu.T, thetah).T;

    wh_x_de[wh_x_de < 0] = 0

    SEP = abs(1-mean(wh_x_nu));

    return (SEP)

def sigma_list(x_nu, x_de):
     
    #print("x_nu, x_de", x_nu, x_de)
    x = np.c_[x_nu, x_de];
    med = compmedDist(x.T);
    # med = 2.5
    return med * array([0.6, 0.8, 1, 1.2, 1.4]);

def lambda_list():
    return 10.0 ** array([-3, -2, -1, 0, 1]);

def norm_pdf(x, mu, std):
    return exp(-(x - mu) ** 2 / (2 * (std ** 2))) / (std * sqrt(2 * pi))

def evaluation(alarm, CP, delta, Seconds):
    N_D_CP = sum(alarm)
    N_A_CP = sum(CP)

    TP = 0
    Used = []

    for i in range(len(CP)):
        TrueCP = 0
        if (CP[i]) > 0:
            j=i
            while (Seconds[i]-Seconds[j]<delta) and j>-1:
                j=j-1

            j=j+1
            while (TrueCP == 0) and (abs(Seconds[j]-Seconds[i])<delta) and (j<len(CP)-1):
                if (alarm[j]) > 0:
                    if j not in Used:
                        TrueCP = 1
                        Used.append(j)
                j = j + 1
            if (TrueCP > 0):
                TP = TP + 1

    FP = N_D_CP - TP
    FN = N_A_CP - TP
    TN = len(CP) - TP - FP - FN

    MAE = 0
    left = 0
    right = len(alarm) - 1


    for i in range(1, len(alarm) - 1):

        if ((CP[i]) > 0):
            gg1 = 0
            gg2 = 0
            k = 0
            while (gg1 < 1) and (gg2 < 1):
                if i-k>0:
                    if ((alarm[i - k]) > 0) :
                        left = i - k
                        gg1 = 1
                if ((i + k) < len(alarm)):
                    if ((alarm[i + k]) > 0):
                        right = i + k
                        gg2 = 1
                k = k + 1
                MAE=MAE+k-1+gg1*abs(Seconds[i]-Seconds[left])+gg2*abs(Seconds[i]-Seconds[right])

    MAE = MAE / N_A_CP

    return (N_A_CP,TP, TN, FP, FN,MAE)

def CPD_Segmentation(filename,feature,CP,alarm):

    num_enabled_activities = max(feature.y) + 1
    # Setup Worksheet to store learning result
    per_class_performance_index = ['true_positive', 'true_negative', 'false_positive', 'false_negative',
                                   'accuracy', 'misclassification', 'recall', 'false positive rate',
                                   'specificity', 'precision', 'prevalence', 'f-1 measure', 'g-measure']

    overall_performance_index = ['average accuracy', 'weighed accuracy',
                                 'precision (micro)', 'recall (micro)', 'f-1 score (micro)',
                                 'precision (macro)', 'recall (macro)', 'f-1 score (macro)',
                                 'exact matching ratio']
    book = xlwt.Workbook()
    overall_sheet = book.add_sheet('overall')
    overall_list_title = ['dataset', 'Classifier'] + overall_performance_index
    overall_list_row = 0
    for c in range(len(overall_list_title)):
        overall_sheet.write(0, c, str(overall_list_title[c]))

    ###segmentation training feature

    num_samples = feature.x.shape[0]

    ii = 0
    new_features = []
    new_labels = []
    sensor_numbers = (feature.x.shape[1] - 7) / 2

    num_test = num_samples / 3
    test_index = range(num_samples - num_test, num_samples)
    train_index = range(num_samples - num_test)

    prev_alarm_index = ii
    while ii < num_samples - num_test:
        if CP[ii] > 0:
            seg_features = np.append(feature.x[prev_alarm_index][:],
                                     feature.x[ii][feature.x.shape[1] - sensor_numbers:feature.x.shape[1]])

            seg_features[0] = ii - prev_alarm_index  # Number of events
            seg_features[2] = seg_features[4]
            seg_features[3] = feature.x[ii][1] - feature.x[prev_alarm_index][1]  # lastEventSeconds (ii-prev)
            seg_features[4] = feature.x[ii][4] - feature.x[prev_alarm_index][4]  # lastEventHour (ii-prev)

            from collections import Counter

            dominant = []

            for tt in range(prev_alarm_index, ii + 1):
                dominant.append(feature.x[tt][3])

            count = Counter(dominant)
            num_occurance = count.most_common()
            seg_features[5] = num_occurance[0][0]

            for dd in range(sensor_numbers):
                for gg in range(prev_alarm_index, ii, 30):
                    seg_features[6 + dd] = seg_features[6 + dd] + feature.x[gg + 1][6 + dd]

            new_features.append(seg_features)
            new_labels.append(feature.y[ii - 1])
            prev_alarm_index = ii
        ii += 1

    train_features = np.array(new_features)
    train_labels = np.array(new_labels)

    ### Segmentation Test feature
    ii = num_samples - num_test
    new_features = []
    new_labels = []
    prev_alarm_index = ii

    while ii < num_samples:
        if alarm[ii] > 0:
            seg_features = np.append(feature.x[prev_alarm_index][:],
                                     feature.x[ii][feature.x.shape[1] - sensor_numbers:feature.x.shape[1]])

            seg_features[0] = ii - prev_alarm_index  # Number of events
            seg_features[2] = seg_features[4]
            seg_features[3] = feature.x[ii][1] - feature.x[prev_alarm_index][1]  # lastEventSeconds (ii-prev)
            seg_features[4] = feature.x[ii][4] - feature.x[prev_alarm_index][4]  # lastEventHour (ii-prev)

            from collections import Counter

            dominant = []

            for tt in range(prev_alarm_index, ii + 1):
                dominant.append(feature.x[tt][3])

            count = Counter(dominant)
            num_occurance = count.most_common()
            seg_features[5] = num_occurance[0][0]

            for dd in range(sensor_numbers):
                for gg in range(prev_alarm_index, ii, 30):
                    seg_features[6 + dd] = seg_features[6 + dd] + feature.x[gg + 1][6 + dd]

            new_features.append(seg_features)
            new_labels.append(feature.y[ii - 1])
            prev_alarm_index = ii
        ii += 1

    test_features = np.array(new_features)

    '''classifiers = [
        (DecisionTreeClassifier(criterion="entropy"), "Decision Tree"),
        (RandomForestClassifier(n_estimators=90, bootstrap=True, criterion="entropy"), "RF"),
        (ExtraTreesClassifier(n_estimators=20, criterion="entropy"), "ET"),
        (GradientBoostingClassifier(n_estimators=20), "GB"),
        (BaggingClassifier(), "Bagging"),
        (AdaBoostClassifier(learning_rate=0.8), "AdaBoost")
    ]

    classifiers_name = ['Decision Tree', 'RF', 'ET', 'GB ', 'Bagging', 'AdaBoost']'''

    classifiers = [
        (DecisionTreeClassifier(criterion="entropy"), "Decision Tree") ]
    classifiers_name = ['Decision Tree']

    index = 0
    for clf, name in classifiers:
        clf.fit(train_features, train_labels)
        predicted_out = clf.predict(test_features)
        predicted_y = np.zeros(num_test)

        pp = 0
        for ww in range(len(predicted_y)):

            if alarm[ww + num_samples - num_test] < 1:
                predicted_y[ww] = int(predicted_out[pp])
            else:
                if pp < len(predicted_out) - 1:
                    pp = pp + 1
                predicted_y[ww] = int(predicted_out[pp])

        ofile = open(classifiers_name[index] + '_labels.xls', 'w')

        ofile.write("{0} {1}\n".format('Labels','Detected Label'))

        for i in range(len(predicted_y)):
            ofile.write("{0},{1}\n".format(feature.get_activity_by_index(feature.y[i + num_samples - num_test]),feature.get_activity_by_index(predicted_y[i])))
        ofile.close()

        ###Result
        confusion_matrix = get_confusion_matrix(num_classes=num_enabled_activities,
                                                label=feature.y[test_index], predicted=predicted_y)
        (overall_performance, per_class_performance) = \
            get_performance_array(num_classes=num_enabled_activities,
                                  confusion_matrix=confusion_matrix)

        overall_list_row += 1
        overall_sheet.write(overall_list_row, 0, filename)
        overall_sheet.write(overall_list_row, 1, (classifiers_name[index]))
        for c in range(len(overall_performance_index)):
            overall_sheet.write(overall_list_row, c + 2, '%.5f' % overall_performance[c])

        index=index+1

    book.save(filename+'_segmentation_performance.xls')

def get_confusion_matrix(num_classes, label, predicted):
    """
    Get Confusion Matrix
    :type num_classes: int
    :param num_class: Number of classes
    :type label: list
    :param label: Data Labels
    :param predicted: Data Labels predicted by classifier
    :return: Confusion Matrix (num_class by num_class) in numpy.array form
    """
    matrix = np.zeros((num_classes, num_classes))
    for i in range(len(label)):

        matrix[label[i]][predicted[i]] += 1
    return matrix

def get_performance_array(num_classes, confusion_matrix):
    """
    Gets performance array for each class
    0 - True_Positive: number of samples that belong to class and classified correctly
    1 - True_Negative: number of samples that correctly classified as not belonging to class
    2 - False_Positive: number of samples that belong to class and not classified correctMeasure:
    3 - False_Negative: number of samples that do not belong to class but classified as class
    4 - Accuracy: Overall, how often is the classifier correct? (TP + TN) / (TP + TN + FP + FN)
    5 - Misclassification: Overall, how often is it wrong? (FP + FN) / (TP + TN + FP + FN)
    6 - Recall: When it's actually yes, how often does it predict yes? TP / (TP + FN)
    7 - False Positive Rate: When it's actually no, how often does it predict yes? FP / (FP + TN)
    8 - Specificity: When it's actually no, how often does it predict no? TN / (FP + TN)
    9 - Precision: When it predicts yes, how often is it correct? TP / (TP + FP)
    10 - Prevalence: How often does the yes condition actually occur in our sample? Total(class) / Total(samples)
    11 - F(1) Measure: 2 * (precision * recall) / (precision + recall)
    12 - G Measure:  sqrt(precision * recall)

    Gets Overall Performance for the classifier
    0 - Average Accuracy: The average per-class effectiveness of a classifier
    1 - Weighed Accuracy: The average effectiveness of a classifier weighed by prevalence of each class
    2 - Precision (micro): Agreement of the class labels with those of a classifiers if calculated from sums of per-text
                           decision
    3 - Recall (micro): Effectiveness of a classifier to identify class labels if calculated from sums of per-text
                        decisions
    4 - F-Score (micro): Relationship between data's positive labels and those given by a classifier based on a sums of
                         per-text decisions
    5 - Precision (macro): An average per-class agreement of the data class labels with those of a classifiers
    6 - Recall (macro): An average per-class effectiveness of a classifier to identify class labels
    7 - F-Score (micro): Relations between data's positive labels and those given by a classifier based on a per-class
                         average
    8 - Exact Matching Ratio: The average per-text exact classification

    Note: In Multi-class classification, Micro-Precision == Micro-Recall == Micro-FScore == Exact Matching Ratio
    (Multi-class classification: each input is to be classified into one and only one class)

    Reference Document:
    Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks.
    Information Processing and Management, 45, p. 427-437

    :param num_classes: Number of classes
    :param confusion_matrix: Confusion Matrix (numpy array of num_class by num_class)
    :return: tuple (overall, per_class)
    """
    per_class_performance_index = ['true_positive', 'true_negative', 'false_positive', 'false_negative',
                                   'accuracy', 'misclassification', 'recall', 'false positive rate',
                                   'specificity', 'precision', 'prevalence', 'f-1 measure', 'g-measure']

    overall_performance_index = ['average accuracy', 'weighed accuracy',
                                 'precision (micro)', 'recall (micro)', 'f-1 score (micro)',
                                 'precision (macro)', 'recall (macro)', 'f-1 score (macro)',
                                 'exact matching ratio']
    #assert(confusion_matrix.shape[0] == confusion_matrix.shape[1])
    #assert(num_classes == confusion_matrix.shape[0])

    per_class = np.zeros((num_classes, len(per_class_performance_index)), dtype=float)
    overall = np.zeros((len(overall_performance_index),), dtype=float)

    for i in range(confusion_matrix.shape[0]):
        true_positive = confusion_matrix[i][i]
        true_negative = np.sum(confusion_matrix)\
            - np.sum(confusion_matrix[i, :])\
            - np.sum(confusion_matrix[:, i])\
            + confusion_matrix[i][i]
        false_positive = np.sum(confusion_matrix[:, i]) - confusion_matrix[i][i]
        false_negative = np.sum(confusion_matrix[i, :]) - confusion_matrix[i][i]
        # Accuracy: (TP + TN) / (TP + TN + FP + FN)
        per_class_accuracy = (true_positive + true_negative)\
            / (true_positive + true_negative + false_positive + false_negative)
        # Mis-classification: (FP + FN) / (TP + TN + FP + FN)
        per_class_misclassification = (false_positive + false_negative)\
            / (true_positive + true_negative + false_positive + false_negative)
        # Recall: TP / (TP + FN)
        if true_positive + false_negative == 0:
            per_class_recall = 0.
        else:
            per_class_recall = true_positive / (true_positive + false_negative)
        # False Positive Rate: FP / (FP + TN)
        if false_positive + true_negative == 0:
            per_class_fpr = 0.
        else:
            per_class_fpr = false_positive / (false_positive + true_negative)
        # Specificity: TN / (FP + TN)
        if false_positive + true_negative == 0:
            per_class_specificity = 0.
        else:
            per_class_specificity = true_negative / (false_positive + true_negative)
        # Precision: TP / (TP + FP)
        if true_positive + false_positive == 0:
            per_class_precision = 0.
        else:
            per_class_precision = true_positive / (true_positive + false_positive)
        # prevalence
        per_class_prevalence = (true_positive + false_negative)\
            / (true_positive + true_negative + false_positive + false_negative)
        # F-1 Measure: 2 * (precision * recall) / (precision +
        if per_class_precision + per_class_recall == 0:
            per_class_fscore = 0.
        else:
            per_class_fscore = 2 * (per_class_precision * per_class_recall) / (per_class_precision + per_class_recall)
        # G Measure: sqrt(precision * recall)
        per_class_gscore = np.sqrt(per_class_precision * per_class_recall)
        per_class[i][0] = true_positive
        per_class[i][1] = true_negative
        per_class[i][2] = false_positive
        per_class[i][3] = false_negative
        per_class[i][4] = per_class_accuracy
        per_class[i][5] = per_class_misclassification
        per_class[i][6] = per_class_recall
        per_class[i][7] = per_class_fpr
        per_class[i][8] = per_class_specificity
        per_class[i][9] = per_class_precision
        per_class[i][10] = per_class_prevalence
        per_class[i][11] = per_class_fscore
        per_class[i][12] = per_class_gscore

    # Average Accuracy: Sum{i}{Accuracy{i}} / num_class
    overall[0] = np.sum(per_class[:, per_class_performance_index.index('accuracy')]) / num_classes
    # Weighed Accuracy: Sum{i}{Accuracy{i} * Prevalence{i}} / num_class
    overall[1] = np.dot(per_class[:, per_class_performance_index.index('accuracy')],
                        per_class[:, per_class_performance_index.index('prevalence')])
    # Precision (micro): Sum{i}{TP_i} / Sum{i}{TP_i + FP_i}
    overall[2] = np.sum(per_class[:, per_class_performance_index.index('true_positive')]) / \
                 np.sum(per_class[:, per_class_performance_index.index('true_positive')] +
                        per_class[:, per_class_performance_index.index('false_positive')])
    # Recall (micro): Sum{i}{TP_i} / Sum{i}{TP_i + FN_i}
    overall[3] = np.sum(per_class[:, per_class_performance_index.index('true_positive')]) / \
                 np.sum(per_class[:, per_class_performance_index.index('true_positive')] +
                        per_class[:, per_class_performance_index.index('false_negative')])
    # F_Score (micro): 2 * Precision_micro * Recall_micro / (Precision_micro + Recall_micro)
    overall[4] = 2 * overall[2] * overall[3] / (overall[2] + overall[3])
    # Precision (macro): Sum{i}{Precision_i} / num_class
    overall[5] = np.sum(per_class[:, per_class_performance_index.index('precision')]) / num_classes
    # Recall (macro): Sum{i}{Recall_i} / num_class
    overall[6] = np.sum(per_class[:, per_class_performance_index.index('recall')]) / num_classes
    # F_Score (macro): 2 * Precision_macro * Recall_macro / (Precision_macro + Recall_macro)
    overall[7] = 2 * overall[5] * overall[6] / (overall[5] + overall[6])
    # Exact Matching Ratio:
    overall[8] = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    return overall, per_class

