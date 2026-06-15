class FeatureSelector:

    def __init__(self,df,target_column,correlation_threshold=0.05):

        self.df = df

        self.target_column = target_column

        self.correlation_threshold = (correlation_threshold)

    def correlation_features(self):

        corr_matrix = self.df.corr(numeric_only=True)

        if (self.target_column not in corr_matrix.columns):

            return []

        target_corr = corr_matrix[self.target_column].abs()

        selected_features = []

        for column, value in (target_corr.items()):

            if (column != self.target_column and value >= self.correlation_threshold):

                selected_features.append(column)

        return selected_features

    def covariance_features(self):

        cov_matrix = self.df.cov(numeric_only=True)

        if (self.target_column not in cov_matrix.columns):

            return []

        target_cov = cov_matrix[self.target_column].abs()

        mean_covariance = (target_cov.mean())

        selected_features = []

        for column, value in (target_cov.items()):

            if (column != self.target_column and value > mean_covariance):

                selected_features.append(column)

        return selected_features

    def select_features(self):
        corr_features = set(self.correlation_features())

        cov_features = set(self.covariance_features())

        selected_features = list(corr_features.union(cov_features))

        if len(selected_features) == 0:

            selected_features = [col for col in self.df.columns if col != self.target_column]

        print(
            "\nNo features selected using "
            "correlation/covariance.")

        print("Using all available features.")

        return selected_features