import pandas as pd

class TitanicDataSet:

    def __init__(self, train_file=None, test_file=None):

        self.train = self.preprocess_data(train_file)
        self.test = self.preprocess_data(test_file, is_test=True)
        self.lastTestIndexProcessed=0

    def preprocess_data(self, path, is_test=False):
        data = pd.read_csv(path, index_col='PassengerId')
        data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
        if is_test:
            data = data.replace([None], [0])
        else:
            data = data[pd.notnull(data['Age'])]
            data = data[pd.notnull(data['Embarked'])]
        data.replace(["female", "male"], [0, 1], inplace=True)
        data.replace(["Q", "C", "S"], [0, 1, 2], inplace=True)
        if "Survived" in data:
            data = data[pd.notnull(data['Survived'])]
        data_norm = (data - data.mean()) / (data.max() - data.min())
        return data_norm

    def next_batch(self,num=None, is_test=False):
        if num is None:
            start = 0
            end = self.train.shape[0]
        else:
            start = self.lastTestIndexProcessed
            end = (self.lastTestIndexProcessed + num) % self.train.shape[0]
        result = self.train[start:end]
        batch_ys = pd.get_dummies(result.pop('Survived').values).as_matrix()
        batch_xs = result.as_matrix()
        return batch_xs, batch_ys
    
