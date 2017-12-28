__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 16 December 2017'
__copyright__ = 'Copyright (c)  2017 Viktor Kerkez'

import abc


class Metrics(object):
    def __init__(self, train_set=None, eval_set=None, test_set=None):

        config = dg.Config()
        self.n_users = config.meta['n_users']
        self.n_movies = config.meta['n_movies']

        dataset = pd.read_csv(dataset)
        cooc = build_cooc_matrix(dataset, self.n_users, self.n_movies)

        # Binarize the cooc
        bin_cooc = cooc.copy()
        bin_cooc[cooc < min_rating] = 0
        bin_cooc[cooc >= min_rating] = 1

        self.true_values = cooc.ravel()
        self.true_values_bin = bin_cooc.ravel()

    @abc.abstractmethod
    def evaluate(self, model):
        # Metrics dictionary
        metrics = {'mae': 0, 'rmse': 0, 'p@k': 0, 'r@k': 0, 'f1': 0, 'auc': 0}

        features = pd.DataFrame({
            'users': np.array(
                [np.arange(self.n_users)] * self.n_movies
            ).T.ravel(),
            'movies': np.array(
                [np.arange(self.n_movies)] * self.n_users
            ).ravel()
        })

        scores = model.predict(features)
        metrics['mae'] = mean_absolute_error(self.true_values, scores)
        metrics['rmse'] = np.sqrt(mean_squared_error(self.true_values, scores))
        for user in range(self.n_users):
            true_values_slice = self.true_values_bin[
                user * self.n_movies:(user + 1) * self.n_movies
            ]
            scores_slice = scores[
                user * self.n_movies:(user + 1) * self.n_movies
            ]
            p, r, f1 = __precision_recall_f1(true_values_slice, scores_slice)
            metrics['p@k'] += p
            metrics['r@k'] += r
            metrics['f1'] += f1
            try:
                metrics['auc'] += roc_auc_score(true_values_slice,
                                                scores_slice)
            except ValueError:
                # This happens in case the user hasn't rated any movies
                pass
        metrics['p@k'] /= self.n_users
        metrics['r@k'] /= self.n_users
        metrics['f1'] /= self.n_users
        metrics['auc'] /= self.n_users
        # Normalize
        return metrics