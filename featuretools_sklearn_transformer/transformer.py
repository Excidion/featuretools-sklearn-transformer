from featuretools import selection, encode_features
from featuretools.computational_backends import calculate_feature_matrix
from featuretools.synthesis import dfs
from sklearn.base import TransformerMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.exceptions import NotFittedError


class DFSTransformer(TransformerMixin):
    """Transformer using Scikit-Learn interface for Pipeline uses.
    """

    def __init__(self,
                 target_dataframe_name=None,
                 agg_primitives=None,
                 trans_primitives=None,
                 allowed_paths=None,
                 max_depth=2,
                 ignore_dataframes=None,
                 ignore_columns=None,
                 seed_features=None,
                 drop_contains=None,
                 drop_exact=None,
                 where_primitives=None,
                 max_features=-1,
                 verbose=False):
        """Creates Transformer

        Args:

            target_dataframe_name (str): Name of dataframe on which to make
                predictions.

            agg_primitives (list[str or AggregationPrimitive], optional): List
                of Aggregation Feature types to apply.

                    Default: ["sum", "std", "max", "skew", "min", "mean",
                              "count", "percent_true", "num_unique", "mode"]

            trans_primitives (list[str or TransformPrimitive], optional):
                List of Transform Feature functions to apply.

                    Default: ["day", "year", "month", "weekday", "haversine",
                              "num_words", "num_characters"]

            allowed_paths (list[list[str]]): Allowed dataframe paths on which to
                make features.

            max_depth (int) : Maximum allowed depth of features.

            ignore_dataframes (list[str], optional): List of dataframes to
                blacklist when creating features.

            ignore_columns (dict[str -> list[str]], optional): List of
                specific columns within each dataframe to blacklist when
                creating features.

            seed_features (list[:class:`.FeatureBase`]): List of manually
                defined features to use.

            drop_contains (list[str], optional): Drop features
                that contains these strings in name.

            drop_exact (list[str], optional): Drop features that
                exactly match these strings in name.

            where_primitives (list[str or PrimitiveBase], optional):
                List of Primitives names (or types) to apply with where
                clauses.

                    Default:

                        ["count"]

            max_features (int, optional) : Cap the number of generated features
                    to this number. If -1, no limit.

        Example:
            .. ipython:: python

                import featuretools as ft
                import pandas as pd

                from featuretools.wrappers import DFSTransformer
                from sklearn.pipeline import Pipeline
                from sklearn.ensemble import ExtraTreesClassifier

                # Get example data
                train_es = ft.demo.load_mock_customer(return_entityset=True, n_customers=3)
                test_es = ft.demo.load_mock_customer(return_entityset=True, n_customers=2)
                y = [True, False, True]

                # Build pipeline
                pipeline = Pipeline(steps=[
                    ('ft', DFSTransformer(target_dataframe_name="customers",
                                          max_features=2)),
                    ('et', ExtraTreesClassifier(n_estimators=100))
                ])

                # Fit and predict
                pipeline.fit(X=train_es, y=y) # fit on customers in training entityset
                pipeline.predict_proba(test_es) # predict probability of each class on test entityset
                pipeline.predict(test_es) # predict on test entityset

                # Same as above, but using cutoff times
                train_ct = pd.DataFrame()
                train_ct['customer_id'] = [1, 2, 3]
                train_ct['time'] = pd.to_datetime(['2014-1-1 04:00',
                                                   '2014-1-2 17:20',
                                                   '2014-1-4 09:53'])

                pipeline.fit(X=(train_es, train_ct), y=y)

                test_ct = pd.DataFrame()
                test_ct['customer_id'] = [1, 2]
                test_ct['time'] = pd.to_datetime(['2014-1-4 13:48',
                                                  '2014-1-5 15:32'])
                pipeline.predict_proba((test_es, test_ct))
                pipeline.predict((test_es, test_ct))

        """
        self.feature_defs = []
        self.target_dataframe_name = target_dataframe_name
        self.agg_primitives = agg_primitives
        self.trans_primitives = trans_primitives
        self.allowed_paths = allowed_paths
        self.max_depth = max_depth
        self.ignore_dataframes = ignore_dataframes
        self.ignore_columns = ignore_columns
        self.seed_features = seed_features
        self.drop_contains = drop_contains
        self.drop_exact = drop_exact
        self.where_primitives = where_primitives
        self.max_features = max_features
        self.verbose = verbose

    def fit(self, X, y=None):
        """Wrapper for DFS

            Calculates a list of features given a dictionary of dataframes and a list
            of relationships. Alternatively, an EntitySet can be passed instead of
            the dataframes and relationships.

            Args:
                X: (ft.Entityset or tuple): Entityset to calculate features on. If a tuple is
                    passed it can take one of these forms: (entityset, cutoff_time_dataframe),
                    (dataframes, relationships), or ((dataframes, relationships), cutoff_time_dataframe)
                y: (iterable): Training targets

            See Also:
                :func:`synthesis.dfs`
        """
        es, dataframes, relationships, _ = parse_x_input(X)

        self.feature_defs = dfs(entityset=es,
                                dataframes=dataframes,
                                relationships=relationships,
                                target_dataframe_name=self.target_dataframe_name,
                                agg_primitives=self.agg_primitives,
                                trans_primitives=self.trans_primitives,
                                allowed_paths=self.allowed_paths,
                                max_depth=self.max_depth,
                                ignore_dataframes=self.ignore_dataframes,
                                ignore_columns=self.ignore_columns,
                                seed_features=self.seed_features,
                                drop_contains=self.drop_contains,
                                drop_exact=self.drop_exact,
                                where_primitives=self.where_primitives,
                                max_features=self.max_features,
                                features_only=True,
                                verbose=self.verbose)

        return self

    def transform(self, X):
        """Wrapper for calculate_feature_matrix

            Calculates a feature matrix for a the given input data and calculation times.

            Args:
                X: (ft.Entityset or tuple): Entityset to calculate features on. If a tuple is
                    passed it can take one of these forms: (entityset, cutoff_time_dataframe),
                    (dataframes, relationships), or ((dataframes, relationships), cutoff_time_dataframe)

            See Also:
                :func:`computational_backends.calculate_feature_matrix`
        """
        es, dataframes, relationships, cutoff_time = parse_x_input(X)

        X_transformed = calculate_feature_matrix(
            features=self.feature_defs,
            instance_ids=None,
            cutoff_time=cutoff_time,
            entityset=es,
            dataframes=dataframes,
            relationships=relationships,
            verbose=self.verbose)

        return X_transformed

    def get_params(self, deep=True):
        out = {
            'target_dataframe_name': self.target_dataframe_name,
            'agg_primitives': self.agg_primitives,
            'trans_primitives': self.trans_primitives,
            'allowed_paths': self.allowed_paths,
            'max_depth': self.max_depth,
            'ignore_dataframes': self.ignore_dataframes,
            'ignore_columns': self.ignore_columns,
            'seed_features': self.seed_features,
            'drop_contains': self.drop_contains,
            'drop_exact': self.drop_exact,
            'where_primitives': self.where_primitives,
            'max_features': self.max_features,
            'verbose': self.verbose,
        }
        return out


def parse_x_input(X):
    if isinstance(X, tuple):
        if isinstance(X[0], tuple):
            # Input of ((dataframes, relationships), cutoff_time)
            dataframes = X[0][0]
            relationships = X[0][1]
            es = None
            cutoff_time = X[1]
        elif isinstance(X[0], dict):
            # Input of (dataframes, relationships)
            dataframes = X[0]
            relationships = X[1]
            es = None
            cutoff_time = None
        else:
            # Input of (entityset, cutoff_time)
            es = X[0]
            dataframes = None
            relationships = None
            cutoff_time = X[1]
    else:
        # Input of entityset
        es = X
        dataframes = None
        relationships = None
        cutoff_time = None

    return es, dataframes, relationships, cutoff_time


class DFSSelectionTransformer(DFSTransformer):
    """Deep Feature Sythesis and Feature Selection Transformer using Scikit-Learn interface for Pipeline usage."""

    def __init__(
        self,
        null_threshold=1.0,
        remove_single_value_features=True,
        count_null_as_value=True,
        **dfs_kwargs,
    ):
        """Creates a Transformer for feature generation.
        Will reduce the amount of generated features with some simple heuristics.
        The reduced featureset overwrites the list of originally generated features.
        This saves compute when calling `.transform()`.

        Args:
            null_threshold (float): If ratio of NULL values exceeds this value, that feature will be removed.
            remove_single_value_features (bool): Remove features with cardinality of 1.
            count_null_as_value (bool): Treat NULL values as a unique value when removing single value features.
            **dfs_kwargs (dict): Keyword arguments passed to `DFSTransformer`

        See Also:
            :class: `DFSTransformer`
            :function: `featuretools.selection.remove_highly_null_features`
            :function: `featuretools.selection.remove_single_value_features`
        """
        super().__init__(**dfs_kwargs)
        if (null_threshold == 1) and not remove_single_value_features:
            ValueError(
                "Set either 'null_threshold < 1' or 'remove_single_value_features = True'."
            )
        self.null_threshold = null_threshold
        self.remove_single_value_features = remove_single_value_features
        self.count_null_as_value = count_null_as_value

    def fit(self, X, y=None):
        """Wrapper for DFS plus and additional feature selection process.

        Calculates a list of features given a dictionary of dataframes and a list
        of relationships. Alternatively, an EntitySet can be passed instead of
        the dataframes and relationships.

        This method constructs the features and fits the supplied selector.
        Optionally some heuristics are used before the selector is fitted.

        Args:
            X: (ft.Entityset or tuple): Entityset to calculate features on. If a tuple is
                passed it can take one of these forms: (entityset, cutoff_time_dataframe),
                (dataframes, relationships), or ((dataframes, relationships), cutoff_time_dataframe)
            y: (iterable): Training targets

        See Also:
            :func:`synthesis.dfs`
        """
        # don't call 'fit_transfrom' to avoid recursion
        super().fit(X, y)
        X = self.transform(X)
        # apply selection heuristics
        if self.null_threshold < 1.0:
            X, self.feature_defs = selection.remove_highly_null_features(
                feature_matrix=X,
                features=self.feature_defs,
                pct_null_threshold=self.null_threshold,
            )
        if self.remove_single_value_features:
            X, self.feature_defs = selection.remove_single_value_features(
                feature_matrix=X,
                features=self.feature_defs,
                count_nan_as_value=self.count_null_as_value,
            )
        if len(self.feature_defs) == 0:
            raise ValueError("Feature selection eliminated all features.")
        return self

    def get_feature_names_out(self):
        if len(self.feature_defs) > 0:
            return [f.generate_name() for f in self.feature_defs]
        else:
            raise NotFittedError


class DFSSelectorTransformer(DFSSelectionTransformer):
    """Deep Feature Sythesis and Feature Selection Transformer using Scikit-Learn interface for Pipeline usage."""

    def __init__(
        self,
        selector,
        encode_categorical=True,
        top_n_categories=10,
        encode_ordinal=False,
        **kwargs,
    ):
        """Creates a Transformer for feature generation.
        Will reduce the amount of generated features with the supplied `selector` and (optionally) some heuristics.
        The reduced featureset overwrites the list of originally generated features.
        This saves compute when calling `.transform()`.

        Args:
            selector (sklearn.selector.SelectorMixin): Sklearn selector used for feature selection.
            encode_categorical (bool): One-hot-encode categorical features.
            top_n_categories (int): Number of top categories per categorical feature to include.
            encode_ordinal (bool): One-hot-encode ordinal categorical features.
            **kwargs (dict): Keyword arguments passed to `DFSSelectionTransformer`

        See Also:
            :class: `DFSSelectionTransformer`
            :class: `sklearn.feature_selection.SelectorMixin`
        """
        super().__init__(**kwargs)
        assert isinstance(
            selector, SelectorMixin
        ), "'selector' has to be of type 'SelectorMixin'"
        self.selector = selector
        self.encode_categorical = encode_categorical
        self.top_n_categories = top_n_categories
        if encode_ordinal:
            assert encode_categorical, "You can only set 'encode_ordinal = True' if also 'encode_categorical = True'."
        self.encode_ordinal = encode_ordinal

    def fit(self, X, y=None):
        """Wrapper for DFS plus and additional feature selection process.

        Calculates a list of features given a dictionary of dataframes and a list
        of relationships. Alternatively, an EntitySet can be passed instead of
        the dataframes and relationships.

        This method constructs the features and fits the supplied selector.
        Optionally some heuristics are used before the selector is fitted.

        Args:
            X: (ft.Entityset or tuple): Entityset to calculate features on. If a tuple is
                passed it can take one of these forms: (entityset, cutoff_time_dataframe),
                (dataframes, relationships), or ((dataframes, relationships), cutoff_time_dataframe)
            y: (iterable): Training targets

        See Also:
            :func:`synthesis.dfs`
        """
        # don't call 'fit_transfrom' to avoid recursion
        super().fit(X, y)
        X = super().transform(X)
        # handle categorical
        if not self.encode_ordinal:
            to_encode = [
                f.generate_name()
                for f in self.feature_defs
                if f.primitive.return_type is None  # seems to be only for categorical
            ]
        else:
            to_encode = None  # encode everything
        if self.encode_categorical:
            X.ww.init()  # without this the encoding fails
            X, self.feature_defs = encode_features(
                feature_matrix=X,
                features=self.feature_defs,
                to_encode=to_encode,
                top_n=self.top_n_categories,
                include_unknown=True,
                drop_first=False,
            )
        # apply selector
        self.selector.fit(X, y)
        mask = self.selector.get_support()
        assert len(mask) == len(self.feature_defs)
        self.feature_defs = [f for f, keep in zip(self.feature_defs, mask) if keep]
        if len(self.feature_defs) == 0:
            raise ValueError("Feature selection eliminated all features.")
        return self

    def get_feature_names_out(self):
        return self.selector.get_feature_names_out()
