import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from scipy.sparse import hstack, csr_matrix

try:

    my_old_data_set = pd.read_csv(r'/Users/ahmedhossam/Desktop/pythonprojects/assigment12/pythonProject1/assets/assets/loan_old.csv')
    # Printing the data
    print(my_old_data_set)
    my_old_data_set.dropna(inplace=True)

    # -----------------------------------------------------------------------------------------------
    # ividing the csv to features and targets
    features = my_old_data_set.iloc[:, 1:-2].values
    targets = my_old_data_set.iloc[:, -2:].values
    print("features after remove",features)
    # -----------------------------------------------------------------------------------------------
    print(' -----------------------------------------------------------------------------------------------')

    # Check for missing values in the whole data
    missing_values = my_old_data_set.isnull().sum()
    print("Number of missing values in each column:")
    print(missing_values)
    print(' -----------------------------------------------------------------------------------------------')


    # function to check every type of every feature
    def check_features_type(features):
        feature_types = {}
        for i in range(features.shape[1]):
            if isinstance(features[0, i], str) or len(np.unique(features[:, i])) <= 5:
                feature_types[i] = 'categorical'
            else:
                feature_types[i] = 'numerical'
        return feature_types

    my_feature_types = check_features_type(features)
    print("Feature types:")
    for feature, ftype in my_feature_types.items():
      print(f"Column {feature}: {ftype}")

    # -----------------------------------------------------------------------------------------------
    # extract nummrical feature
    def extract_numerical_features(features, feature_types):

        numerical_features_indices = [i for i, ftype in feature_types.items() if ftype == 'numerical']
        numerical_features = features[:, numerical_features_indices]
        return numerical_features


    numerical_features = extract_numerical_features(features, my_feature_types)
    summary_stats = np.nanpercentile(numerical_features,[0, 50, 100], axis=0)
    print("Summary st for numerical features:", summary_stats)
    #----------------------------------------------------------------------------------------------
    def extract_numerical_features_indexes(features, my_feature_types):

        numerical_features_indices = [i for i, ftype in my_feature_types.items() if ftype == 'numerical']
        return numerical_features_indices


    numerical_features_indices=extract_numerical_features_indexes(features, my_feature_types)


    def extract_categorical_features(features, my_feature_types):
        categorical_features_indices = [i for i, ftype in my_feature_types.items() if ftype == 'categorical']
        return categorical_features_indices


    categorical_features_indices=extract_categorical_features(features,my_feature_types)



    # -----------------------------------------------------------------------------------------------
    numerical_features_df = pd.DataFrame(numerical_features)
    sns.pairplot(numerical_features_df)
    plt.show()


# -----------------------------------------------------------------------------------------------
    my_old_data_set.dropna(inplace=True)


    # -----------------------------------------------------------------------------------------------

    # Shuffle and split the data into training and testing sets
    features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size=0.2,random_state=1)
    print("featuretrain",features_train)



    # -----------------------------------------------------------------------------------------------
    def encode_categorical_features(features_train, features_test):
        # Apply one-hot encoding to specified columns
        one_hot_indices = [2, 8]
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        one_hot_features_train_encoded = one_hot_encoder.fit_transform(features_train[:, one_hot_indices])
        one_hot_features_test_encoded = one_hot_encoder.fit_transform(features_test[:, one_hot_indices])

        # Extract column names from one-hot encoder and rename them
        one_hot_column_names = one_hot_encoder.get_feature_names_out(input_features=["Dependents", "Property_Area"])
        one_hot_train_df = pd.DataFrame(one_hot_features_train_encoded.toarray(), columns=one_hot_column_names)
        one_hot_test_df = pd.DataFrame(one_hot_features_test_encoded.toarray(), columns=one_hot_column_names)

        # Apply label encoding to specified columns
        label_indices = [0, 1, 3, 7]
        label_features_train_encoded = pd.DataFrame()
        label_features_test_encoded = pd.DataFrame()
        for idx in label_indices:

            label_encoder = LabelEncoder()
            label_encoded_train = label_encoder.fit_transform(features_train[:, idx])
            label_features_train_encoded["label_" + str(idx)] = label_encoded_train

            label_encoded_test = label_encoder.transform(features_test[:, idx])
            label_features_test_encoded["label_" + str(idx)] = label_encoded_test

        features_train_df = pd.concat([one_hot_train_df, label_features_train_encoded], axis=1)
        features_test_df = pd.concat([one_hot_test_df, label_features_test_encoded], axis=1)

        print("Training Set:")
        print(features_train_df)
        print("\nTest Set:")
        print(features_test_df)

        return features_train_df, features_test_df


    features_train_encoded, features_test_encoded = encode_categorical_features(features_train, features_test)

    print("features encoded")
    print(features_train_encoded)


    #------------------------------------------------------------------------------------------------------------
    # encode my targets

    def encode_categorical_targets(targets_train, targets_test):

        # Extract the binary labels from the second column of the target arrays
        labels_train = targets_train[:, 1].reshape(-1, 1)
        labels_test = targets_test[:, 1].reshape(-1, 1)

        # Initialize label encoder
        label_encoder = LabelEncoder()

        # Fit label encoder on the binary labels and transform them
        encoded_labels_train = label_encoder.fit_transform(labels_train.flatten())
        encoded_labels_test = label_encoder.transform(labels_test.flatten())

        # Replace the second column in the original target with the encoded binary labels
        targets_train_encoded = targets_train.copy()
        targets_train_encoded[:, 1] = encoded_labels_train.reshape(-1)

        targets_test_encoded = targets_test.copy()
        targets_test_encoded[:, 1] = encoded_labels_test.reshape(-1)

        return targets_train_encoded, targets_test_encoded


    targets_train_encoded, targets_test_encoded = encode_categorical_targets(targets_train, targets_test)
    print('encode targets', targets_train_encoded)
    #------------------------------------------------------------------------------------------------------------
    # standlize my data

    def standardize_numerical_features(features_train, features_test, numerical_features_indices):
        # Initialize the StandardScaler
        scaler = StandardScaler()

        # Fit the scaler on the training data and transform both training and test data
        features_train[:, numerical_features_indices] = scaler.fit_transform(
            features_train[:, numerical_features_indices])
        features_test[:, numerical_features_indices] = scaler.fit_transform(features_test[:, numerical_features_indices])

        return features_train, features_test


    features_train_scaled, features_test_scaled = standardize_numerical_features(features_train,
                                                                                 features_test,
                                                                                 numerical_features_indices)
    print("data standlized ", features_train_scaled)
    categorical_indices = [0, 1, 2, 3, 7, 8]

    # Remove the categorical columns
    features_train_scaled_numeric = np.delete(features_train_scaled, categorical_indices, axis=1)
    features_test_scaled_numeric = np.delete(features_test_scaled, categorical_indices, axis=1)

    # Concatenate encoded categorical features with numerical features
    features_train_final = np.concatenate((features_train_scaled_numeric, features_train_encoded), axis=1)
    features_test_final = np.concatenate((features_test_scaled_numeric, features_test_encoded), axis=1)

    print("Final Features Train:modifiucation")
    print(features_train_final)

    print("\nFinal Features Test:modifiucation")
    print(features_test_final)
    # ------------------------------------------------------------------------------------------------------------
    #implement linear regression
    def linear_regression(features_train_final,targets_train_encoded, features_test_final,targets_test_encoded):
        #CREATE OBJECT FROM MY MODEL
        linear_regression_model = LinearRegression()

        # Fit the model to the training data
        linear_regression_model.fit(features_train_final, targets_train_encoded[:,0])
        predictions_test = linear_regression_model.predict(features_test_final)

        # Calculate R-squared score for training and testing data
        r2_test = r2_score(targets_test_encoded[:,0], predictions_test)

        return linear_regression_model, r2_test


    linear_regression_model,r2_test = linear_regression(features_train_final,
                                                                                      targets_train_encoded,
                                                                                      features_test_final,
                                                                                      targets_test_encoded)

    print("Testing R-squared score:", r2_test)
    #------------------------------------------------------------------------------------------------------------

    # logistic model hamada
    # hamda(0.01,11)
    # hamda.fit(features,targets)
    class LogisticRegression:
        def __init__(self, learning_rate=0.01, n_iters=10000):
            self.lr = learning_rate
            self.iterations = n_iters
            self.weights = None
            self.bias = None

        def fit(self, X, y):
            n_samples, n_features = X.shape

            # Initialize parameters
            self.weights = np.zeros(n_features)
            self.bias = 0  ### w1 x1+w2 x2 +w3 x3 ------ b

            # Gradient descent
            for _ in range(self.iterations):
                self._gradient_descent(X, y, n_samples)

        def _gradient_descent(self, X, y, n_samples):
            # Compute predictions
            linear_model = X.dot(self.weights) + self.bias

            linear_model_float = linear_model.astype(float)

            y_predicted = self._sigmoid(linear_model_float)

            y_predicted_float = y_predicted.astype(float)



            y_first_column = y[:, 0].astype(float)

            error = y_predicted_float - y_first_column

            dw = (1 / n_samples) * X.T.dot(error).astype(float)

            db = (1 / n_samples) * np.sum(error).astype(float)


            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        def predict(self, X):
            linear_model = X.dot(self.weights) + self.bias.reshape(1, -1)

            linear_model_float = linear_model.astype(float)
            y_predicted = self._sigmoid(linear_model_float)

            y_predicted_cls = np.where(y_predicted > 0.5, 1, 0)

            return y_predicted_cls

        def _sigmoid(self, x):

            return 1.0 / (1.0 + np.exp(-x))

    def logistic_accuracy(y_true, y_pred):

        accuracy = np.sum(y_true[:, 1] == y_pred) / len(y_true)

        return accuracy

    logistic_model = LogisticRegression()
    logistic_model.fit(features_train_final, targets_train_encoded)
    predictions = logistic_model.predict(features_test_final)
    accuracy = logistic_accuracy(targets_test_encoded, predictions)
    print("Accuracy:", accuracy)
    my_new_data_set = pd.read_csv('/Users/ahmedhossam/Desktop/pythonprojects/assigment12/pythonProject1/assets/assets/loan_new.csv')
    my_new_data_set = my_new_data_set.drop(my_old_data_set.columns[0], axis=1)
    my_new_data_set.dropna(inplace=True)
    print(my_new_data_set)


    def preprocess_data(my_new_data_set, numerical_features_indices):

        def encode_categorical_features(my_new_data_set):

            # Apply one-hot encoding to specified columns
            one_hot_columns = ['Dependents', 'Property_Area']
            one_hot_data = my_new_data_set[one_hot_columns]

            # Apply one-hot encoding
            one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
            one_hot_features_encoded = one_hot_encoder.fit_transform(one_hot_data)

            # Apply label encoding to specified columns
            label_columns = ['Gender', 'Married', 'Education', 'Credit_History']
            label_data = my_new_data_set[label_columns]

            label_features_encoded = []
            label_encoder = LabelEncoder()
            for col in label_columns:

                label_encoded_data = label_encoder.fit_transform(label_data[col])
                label_features_encoded.append(label_encoded_data)

            # Convert lists to numpy arrays
            label_features_encoded = np.array(label_features_encoded).T

            # Concatenate features
            features_encoded = hstack([one_hot_features_encoded, label_features_encoded])
            return features_encoded

        def standardize_numerical_features(my_new_data_set, numerical_features_indices):

            scaler = StandardScaler()

            my_new_data_set[:, numerical_features_indices] = scaler.fit_transform(
                my_new_data_set[:, numerical_features_indices])
            return my_new_data_set

        # Apply encoding and standardization
        features_encoded = encode_categorical_features(my_new_data_set)
        features_scaled = standardize_numerical_features(my_new_data_set, numerical_features_indices)

        # Separate categorical indices
        categorical_indices = [0, 1, 2, 3, 7, 8]
        # Remove the categorical columns from the encoded features
        features_encoded_numeric = features_scaled[:,
                                   ~np.isin(np.arange(features_scaled.shape[1]), categorical_indices)]

        # Concatenate encoded categorical features with numerical features
        features_final = hstack([features_encoded_numeric, features_scaled[:, categorical_indices]])

        return features_final

    features_scaled_predicted = preprocess_data(my_new_data_set, numerical_features_indices)
    print("features",features_scaled_predicted)
    predicted_linear_regression_values=linear_regression_model.predict(features_scaled_predicted)
    print("predicted",predicted_linear_regression_values)
    predicted_logistic_regression_values=logistic_model.predict(features_scaled_predicted)
    print("predicted logistic",predicted_logistic_regression_values)





















except FileNotFoundError:
    print("File not found!")
except PermissionError:
    print("Permission denied!")
except Exception as e:
    print("An error occurred:", e)