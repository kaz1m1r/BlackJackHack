import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import accuracy_score


class BlackJackBot:
    def __init__(self):
        pass

    def knn(self, partitions, predictors, outcome):
        # making individual that's necessary to determine the optimal amount of neighbors
        test_individual = partitions['valid_X'][predictors].iloc[0, :]

        # making initial KNN model
        knn = NearestNeighbors(n_neighbors=3)
        knn.fit(partitions['train_X'][predictors])

        results = []
        for k in range(1, 40):
            knn = KNeighborsClassifier(n_neighbors=k).fit(partitions['train_X'], partitions['train_y'])
            results.append({
                'k': k,
                'accuracy': accuracy_score(predictors['valid_y'], knn.predict(predictors['valid_X']))
            })


    class Partitioner:
        def __init__(self):
            pass

        def GetFourNormalizedPartitions(self, dataset, predictors, outcome, test_size=0.4):
            """
            Method that's used to make four normalized partitions
            :param dataset:
            :param predictors: Names of dataset's columns that are your predictors
            :param outcome: Target column from the dataset
            :param test_size: Size of the normalized test partition
            :return dict:
            """
            d = preprocessing.normalize(dataset, axis=0)
            dataset_norm = pd.DataFrame(d, columns=dataset.columns)
            X = dataset_norm[predictors]
            y = dataset_norm[outcome]

            train_X, valid_X, train_y, valid_y = train_test_split(X, y, random_state=26, test_size=test_size)
            partitions = {
                'train_X': train_X,
                'valid_X': valid_X,
                'train_y': train_y,
                'valid_y': valid_y
            }
            return partitions

if __name__ == "__main__":
    path = "/home/kaz1m1r/PycharmProjects/BlackJackHack/blkjckhands.csv/blkjckhands.csv"
    dataset = pd.read_csv(path)
    cols = [
        'card1',
        'card2',
        'card3',
        'card4',
        'card5',
        'dealcard1',
        'dealcard2',
        'dealcard3',
        'dealcard4',
        'dealcard5',
        'winloss'
    ]
    dataset = pd.get_dummies(dataset[cols], drop_first=True)
    d = preprocessing.normalize(dataset, axis=0)
    dataset_norm = pd.DataFrame(d, columns=dataset.columns)

    predictors = cols[:-1]
    outcome_winloss_Push = 'winloss_Push'
    outcome_winloss_Win = 'winloss_Win'