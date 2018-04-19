from sklearn.metrics import mean_squared_error, average_precision_score, confusion_matrix, accuracy_score, f1_score
from collections import defaultdict
from sklearn.model_selection import KFold
import audiphil as au
import numpy as np

def kfold(X, y, clfs_names, num_splits, cm=False, cm_size=15):
    classifiers = []
    d = defaultdict(list)
    labels = sorted(list(set(y)))
    for sm_c, name in clfs_names:
        scores = []

        kf = KFold(n_splits=num_splits, shuffle=True)

        for train_index, test_index in kf.split(X):
            X_train, X_test = np.array(X[train_index]), np.array(X[test_index])
            y_train, y_test = np.array(y[train_index]), np.array(y[test_index])

            cur = sm_c()
            cur = cur.fit(X_train, y_train)
            y_res = cur.predict(X_test)

            #cur_acc = f1_score(y_test, y_res, average="macro")
            cur_acc = accuracy_score(y_test, y_res)

            if cm:
                mycm = confusion_matrix(y_test, y_res, labels=labels)
                au.plot_confusion_matrix(mycm, classes=labels, normalize=False, title="{} {}".format(name, "%0.2f" % accuracy_score(y_test, y_res)), cm_size=cm_size)

            scores.append(cur_acc)
            classifiers.append(cur)

        scores = np.array(scores)
        d[name].append("KFOLD: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return (d, classifiers)

