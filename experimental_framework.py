from sklearn.model_selection import StratifiedKFold
from scipy import sparse
from skml.datasets import sample_down_label_space
# liac-arff
import arff
import random

random.seed(2018)


def load_from_arff(filename, labelcount, endian="big",
    input_feature_type='float', encode_nominal=True, load_sparse=False,
    return_attribute_definitions=False):
    """Method for loading ARFF files as numpy array
    Parameters
    ----------
    filename : str
        path to ARFF file
    labelcount: integer
        number of labels in the ARFF file
    endian: str {"big", "little"} (default is "big")
        whether the ARFF file contains labels at the beginning of the
        attributes list ("big" endianness, MEKA format) or at the end
        ("little" endianness, MULAN format)
    input_feature_type: numpy.type as string (default is "float")
        the desire type of the contents of the return 'X' array-likes,
        default 'i8', should be a numpy type,
        see http://docs.scipy.org/doc/numpy/user/basics.types.html
    encode_nominal: bool (default is True)
        whether convert categorical data into numeric factors - required
        for some scikit classifiers that can't handle non-numeric
        input features.
    load_sparse: boolean (default is False)
        whether to read arff file as a sparse file format, liac-arff
        breaks if sparse reading is enabled for non-sparse ARFFs.
    return_attribute_definitions: boolean (default is False)
        whether to return the definitions for each attribute in the
        dataset
    Returns
    -------
    X : scipy.sparse
        matrix with :code:`input_feature_type` elements
    y: scipy.sparse
        matrix of binary label indicator matrix
    """
    matrix = None

    if not load_sparse:
        arff_frame = arff.load(open(filename, 'r'),
                               encode_nominal=encode_nominal,
                               return_type=arff.DENSE)
        try:
            matrix = sparse.csr_matrix(
                arff_frame['data'], dtype=input_feature_type)
        except:
            print(arff_frame['data'])
    else:
        arff_frame = arff.load(open(filename, 'r'),
                               encode_nominal=encode_nominal,
                               return_type=arff.COO)
        data = arff_frame['data'][0]
        row = arff_frame['data'][1]
        col = arff_frame['data'][2]
        matrix = sparse.coo_matrix((data, (row, col)),
                                   shape=(max(row) + 1, max(col) + 1))

    X, y = None, None

    if endian == "big":
        X, y = matrix.tocsc()[:, labelcount:].tolil(), matrix.tocsc()[
            :, :labelcount].astype(int).tolil()
    elif endian == "little":
        X, y = matrix.tocsc()[
            :, :-labelcount].tolil(), matrix.tocsc()[:, -labelcount:].astype(int).tolil()
    else:
        # unknown endian
        return None

    if return_attribute_definitions:
        return X, y, arff_frame['attributes']
    else:
        return X, y


def load_data():
    data = {
        # src: MULAN
        'scene': load_from_arff('data/scene/scene.arff',
                                labelcount=6, endian="little"),
        'emotions': load_from_arff('data/emotions/emotions.arff',
                                   labelcount=6, endian="little"),
        'yeast-10': load_from_arff('data/yeast/yeast.arff',
                                   labelcount=14, endian="little"),
        'mediamill-10': load_from_arff('data/mediamill/mediamill.arff',
                                       labelcount=101, endian="little"),
        'enron-10': load_from_arff('data/enron/enron.arff',
                                   labelcount=53, endian="little"),
        'medical-10': load_from_arff('data/medical/medical.arff',
                                     labelcount=44, endian="little"),
        'slashdot-10': load_from_arff('data/slashdot/SLASHDOT-F.arff',
                                      labelcount=22),
        'ohsumed-10': load_from_arff('data/ohsumed/OHSUMED-F.arff',
                                     labelcount=23),
        'tmc2007-500-10': load_from_arff('data/tmc2007-500/tmc2007-500.arff',
                                         labelcount=22, endian="little"),
        # head data/imdb/IMDB-F.arff  -n 40 | grep "{0,1}" | uniq | wc -l
        'imdb-10': load_from_arff('data/imdb/IMDB-F.arff',
                                  labelcount=28)
    }

    for k in data:
        print(k)
        print(data[k])
        if data[k][1].shape[1] > 10:
            data[k] = (data[k][0], sample_down_label_space(data[k][1], 10))

    return data


def k_fold_cv(clf, X, y, k=3, shuffle=True):
    """
    Runs k-fold cross validation on the data given the classifier and parameters.
    Returns a list of predictions with length k, where each item consists of the
    predictions for the i-th fold with $i=1...k$.
    """
    kf = StratifiedKFold(n_splits=k, random_state=random_state, shuffle=shuffle)
    preds = []

    for train_index, test_index in kf.split(X, y):

        # assuming classifier object exists
        X_train = X[train_index,:]
        y_train = y[train_index,:]

        X_test = X[test_index,:]
        y_test = y[test_index,:]

        # learn the classifier
        clf.fit(X_train, y_train)

        # predict labels for test data
        predictions = clf.predict(X_test)
        preds.append(predictions)

    return preds

if __name__ == '__main__':
    data = load_data()

    for k in data:
        print(k, ":", data[k][1].shape)