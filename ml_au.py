
from collections import defaultdict

def kfold(X, y, num_k, clfs_names):

    d = defaultdict(list)
    for sm_c, name in clfs_names:

        scores = []
        kf = KFold(n_splits=num_splits, shuffle=True)

        for train_index, test_index in kf.split(X):
            X_train, X_test = a(X[train_index]), a(X[test_index])
            y_train, y_test = a(y[train_index]), a(y[test_index])

            cur = sm_c()
            cur = cur.fit(X_train, y_train)
            y_res = cur.predict(X_test)

            cur_acc = f1_score(y_test, y_res, average="macro")
            scores.append(cur_acc)

        scores = np.array(scores)
        d[name].append("KFOLD: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return d


