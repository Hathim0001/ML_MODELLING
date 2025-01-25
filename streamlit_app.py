import numpy as np

import pandas as pd

from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC, SVR

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

from sklearn.preprocessing import LabelEncoder

st.set_option("deprecation.showPyplotGlobalUse", False)
import warnings

warnings.filterwarnings("ignore")

def load_dataset(Data):

    if Data == "Iris":
        return datasets.load_iris()
    elif Data == "Wine":
        return datasets.load_wine()
    elif Data == "Breast Cancer":
        return datasets.load_breast_cancer()
    elif Data == "Diabetes":
        return datasets.load_diabetes()
    elif Data == "Digits":
        return datasets.load_digits()
    elif Data == "Salary":
        return pd.read_csv("Dataset/Salary_dataset.csv")
    elif Data == "Naive Bayes Classification":
        return pd.read_csv("Dataset/Naive-Bayes-Classification-Data.csv")
    elif Data == "Heart Disease Classification":
        return pd.read_csv("Dataset/Updated_heart_prediction.csv")
    elif Data == "Titanic":
        return pd.read_csv("Dataset/Preprocessed Titanic Dataset.csv")
    else:
        return pd.read_csv("Dataset/car_evaluation.csv")

def Input_output(data, data_name):

    if data_name == "Salary":
        X, Y = data["YearsExperience"].to_numpy().reshape(-1, 1), data[
            "Salary"
        ].to_numpy().reshape(-1, 1)

    elif data_name == "Naive Bayes Classification":
        X, Y = data.drop("diabetes", axis=1), data["diabetes"]

    elif data_name == "Heart Disease Classification":
        X, Y = data.drop("output", axis=1), data["output"]

    elif data_name == "Titanic":
        X, Y = (
            data.drop(
                columns=["survived", "home.dest", "last_name", "first_name", "title"],
                axis=1,
            ),
            data["survived"],
        )

    elif data_name == "Car Evaluation":

        df = data

        le = LabelEncoder()

        func = lambda i: le.fit(df[i]).transform(df[i])
        for i in df.columns:
            df[i] = func(i)

        X, Y = df.drop(["unacc"], axis=1), df["unacc"]

    else:
        X = data.data
        Y = data.target

    return X, Y

def add_parameter_classifier_general(algorithm):

    params = dict()

    if algorithm == "SVM":

        c_regular = st.sidebar.slider("C (Regularization)", 0.01, 10.0)
        kernel_custom = st.sidebar.selectbox(
            "Kernel", ("linear", "poly ", "rbf", "sigmoid")
        )
        params["C"] = c_regular
        params["kernel"] = kernel_custom

    elif algorithm == "KNN":

        k_n = st.sidebar.slider("Number of Neighbors (K)", 1, 20, key="k_n_slider")
        params["K"] = k_n
        weights_custom = st.sidebar.selectbox("Weights", ("uniform", "distance"))
        params["weights"] = weights_custom

    elif algorithm == "Naive Bayes":
        st.sidebar.info(
            "This is a simple Algorithm. It doesn't have Parameters for Hyper-tuning."
        )

    elif algorithm == "Decision Tree":

        max_depth = st.sidebar.slider("Max Depth", 2, 17)
        criterion = st.sidebar.selectbox("Criterion", ("gini", "entropy"))
        splitter = st.sidebar.selectbox("Splitter", ("best", "random"))
        params["max_depth"] = max_depth
        params["criterion"] = criterion
        params["splitter"] = splitter

        try:
            random = st.sidebar.text_input("Enter Random State")
            params["random_state"] = int(random)
        except:
            params["random_state"] = 4567

    elif algorithm == "Random Forest":

        max_depth = st.sidebar.slider("Max Depth", 2, 17)
        n_estimators = st.sidebar.slider("Number of Estimators", 1, 90)
        criterion = st.sidebar.selectbox("Criterion", ("gini", "entropy", "log_loss"))
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        params["criterion"] = criterion

        try:
            random = st.sidebar.text_input("Enter Random State")
            params["random_state"] = int(random)
        except:
            params["random_state"] = 4567


    # Adding Parameters for Logistic Regression
    else:

        # Adding regularization parameter from range 0.01 to 10.0
        c_regular = st.sidebar.slider("C (Regularization)", 0.01, 10.0)
        params["C"] = c_regular
        # Taking fit_intercept
        fit_intercept = st.sidebar.selectbox("Fit Intercept", ("True", "False"))
        params["fit_intercept"] = bool(fit_intercept)
        # Taking Penalty only l2 and None is supported
        penalty = st.sidebar.selectbox("Penalty", ("l2", None))
        params["penalty"] = penalty
        # Taking n_jobs
        n_jobs = st.sidebar.selectbox("Number of Jobs", (None, -1))
        params["n_jobs"] = n_jobs

    return params


def add_parameter_regressor(algorithm):
    params = dict()

    if algorithm == "Decision Tree":
        max_depth = st.sidebar.slider("Max Depth", 2, 17)
        criterion = st.sidebar.selectbox("Criterion", ("absolute_error", "squared_error", "poisson", "friedman_mse"))
        splitter = st.sidebar.selectbox("Splitter", ("best", "random"))
        params["max_depth"] = max_depth
        params["criterion"] = criterion
        params["splitter"] = splitter
        try:
            random = st.sidebar.text_input("Enter Random State")
            params["random_state"] = int(random)
        except:
            params["random_state"] = 4567

    elif algorithm == "Linear Regression":
        fit_intercept = st.sidebar.selectbox("Fit Intercept", ("True", "False"))
        params["fit_intercept"] = bool(fit_intercept)
        n_jobs = st.sidebar.selectbox("Number of Jobs", (None, -1))
        params["n_jobs"] = n_jobs

    else:
        max_depth = st.sidebar.slider("Max Depth", 2, 17)
        n_estimators = st.sidebar.slider("Number of Estimators", 1, 90)
        criterion = st.sidebar.selectbox("Criterion", ("absolute_error", "squared_error", "poisson", "friedman_mse"))
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        params["criterion"] = criterion
        try:
            random = st.sidebar.text_input("Enter Random State")
            params["random_state"] = int(random)
        except:
            params["random_state"] = 4567

    return params

def model_classifier(algorithm, params):
    if algorithm == "KNN":
        return KNeighborsClassifier(n_neighbors=params["K"], weights=params["weights"])
    elif algorithm == "SVM":
        return SVC(C=params["C"], kernel=params["kernel"])
    elif algorithm == "Decision Tree":
        return DecisionTreeClassifier(criterion=params["criterion"], splitter=params["splitter"], random_state=params["random_state"])
    elif algorithm == "Naive Bayes":
        return GaussianNB()
    elif algorithm == "Random Forest":
        return RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], criterion=params["criterion"], random_state=params["random_state"])
    elif algorithm == "Linear Regression":
        return LinearRegression(fit_intercept=params["fit_intercept"], n_jobs=params["n_jobs"])
    else:
        return LogisticRegression(fit_intercept=params["fit_intercept"], penalty=params["penalty"], C=params["C"], n_jobs=params["n_jobs"])

def model_regressor(algorithm, params):
    if algorithm == "KNN":
        return KNeighborsRegressor(n_neighbors=params["K"], weights=params["weights"])
    elif algorithm == "SVM":
        return SVR(C=params["C"], kernel=params["kernel"])
    elif algorithm == "Decision Tree":
        return DecisionTreeRegressor(criterion=params["criterion"], splitter=params["splitter"], random_state=params["random_state"])
    elif algorithm == "Random Forest":
        return RandomForestRegressor(n_estimators=params["n_estimators"], max_depth=params["max_depth"], criterion=params["criterion"], random_state=params["random_state"])
    else:
        return LinearRegression(fit_intercept=params["fit_intercept"], n_jobs=params["n_jobs"])

def info(data_name, algorithm, algorithm_type, data, X, Y):
    datasets_info = {
        "Diabetes": "Regression",
        "Salary": "Regression",
        "Naive Bayes Classification": "Classification",
        "Car Evaluation": "Classification",
        "Heart Disease Classification": "Classification",
        "Titanic": "Classification",
    }

    target_names_mapping = {
        "Naive Bayes Classification": ["Not Diabetic", "Diabetic"],
        "Heart Disease Classification": ["Less Chance Of Heart Attack", "High Chance Of Heart Attack"],
    }

    st.write(f"## {datasets_info.get(data_name, 'Classification')} {data_name} Dataset")
    st.write(f"Algorithm is: {algorithm} {algorithm_type}")
    st.write("Shape of Dataset is:", X.shape)

    if datasets_info.get(data_name) == "Classification":
        st.write("Number of classes:", len(np.unique(Y)))
        target_names = target_names_mapping.get(data_name, data.target_names if hasattr(data, 'target_names') else [])
        df = pd.DataFrame({"Target Value": list(np.unique(Y)), "Target Name": target_names})
        st.write("Values and Name of Classes")
        st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)


    elif data_name == "Titanic":

        st.write(f"## Classification {data_name} Dataset")
        st.write(f'Algorithm is : {algorithm + " " + algorithm_type}')

        # Printing shape of data
        st.write("Shape of Dataset is: ", X.shape)
        st.write("Number of classes: ", len(np.unique(Y)))
        # Making a dataframe to store target name and value

        df = pd.DataFrame(
            {
                "Target Value": list(np.unique(Y)),
                "Target Name": ["Not Survived", "Survived"],
            }
        )

        # Display the DataFrame without index labels
        st.write("Values and Name of Classes")

        # Display the DataFrame as a Markdown table
        # To successfully run this we need to install tabulate
        st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)
        st.write("\n")

    else:

        st.write(f"## Classification {data_name} Dataset")
        st.write(f"Algorithm is : {algorithm}")

        # Printing shape of data
        st.write("Shape of Dataset is: ", X.shape)
        st.write("Number of classes: ", len(np.unique(Y)))
        # Making a dataframe to store target name and value

        df = pd.DataFrame(
            {
                "Target Value": list(np.unique(Y)),
                "Target Name": [
                    "Unacceptable",
                    "Acceptable",
                    "Good Condition",
                    "Very Good Condition",
                ],
            }
        )

       
        st.write("Values and Name of Classes")

        st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)
        st.write("\n")


def choice_classifier(data, data_name, X, Y):


    if data_name == "Diabetes":
        plt.scatter(X[:, 0], X[:, 1], c=Y, cmap="viridis", alpha=0.8)
        plt.title("Scatter Classification Plot of Dataset")
        plt.colorbar()


    elif data_name == "Digits":
        colors = [
            "purple",
            "green",
            "yellow",
            "red",
            "black",
            "cyan",
            "pink",
            "magenta",
            "grey",
            "teal",
        ]
        sns.scatterplot(
            x=X[:, 0],
            y=X[:, 1],
            hue=Y,
            palette=sns.color_palette(colors),
            cmap="viridis",
            alpha=0.4,
        )
        # Giving legend
        # If we try to show the class target name it will show in different color than the ones that are plotted
        plt.legend(data.target_names, shadow=True)
        # Giving Title
        plt.title("Scatter Classification Plot of Dataset With Target Classes")

    elif data_name == "Salary":
        sns.scatterplot(x=data["YearsExperience"], y=data["Salary"], data=data)
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.title("Scatter Classification Plot of Dataset")

    elif data_name == "Naive Bayes Classification":
        colors = ["purple", "green"]
        sns.scatterplot(
            x=data["glucose"],
            y=data["bloodpressure"],
            data=data,
            hue=Y,
            palette=sns.color_palette(colors),
            alpha=0.4,
        )
        plt.legend(shadow=True)
        plt.xlabel("Glucose")
        plt.ylabel("Blood Pressure")
        plt.title("Scatter Classification Plot of Dataset With Target Classes")

    # We cannot give data directly we have to specify the values for x and y
    else:
        colors = ["purple", "green", "yellow", "red"]
        sns.scatterplot(
            x=X[:, 0], y=X[:, 1], hue=Y, palette=sns.color_palette(colors), alpha=0.4
        )
        plt.legend(shadow=True)
        plt.title("Scatter Classification Plot of Dataset With Target Classes")



def choice_regressor(X, x_test, predict, data, data_name, Y, fig):

    # Plotting Regression Plot for dataset diabetes
    # Since this is a regression dataset we show regression line as well
    if data_name == "Diabetes":
        plt.scatter(X[:, 0], Y, c=Y, cmap="viridis", alpha=0.4)
        plt.plot(x_test, predict, color="red")
        plt.title("Scatter Regression Plot of Dataset")
        plt.legend(["Actual Values", "Best Line or General formula"])
        plt.colorbar()

    elif data_name == "Digits":
        colors = [
            "purple",
            "green",
            "yellow",
            "red",
            "black",
            "cyan",
            "pink",
            "magenta",
            "grey",
            "teal",
        ]
        sns.scatterplot(
            x=X[:, 0],
            y=X[:, 1],
            hue=Y,
            palette=sns.color_palette(colors),
            cmap="viridis",
            alpha=0.4,
        )
        plt.plot(x_test, predict, color="red")
        # Giving legend
        # If we try to show the class target name it will show in different color than the ones that are plotted
        plt.legend(data.target_names, shadow=True)
        # Giving Title
        plt.title("Scatter Plot of Dataset With Target Classes")

    elif data_name == "Salary":
        sns.scatterplot(x=data["YearsExperience"], y=data["Salary"], data=data)
        plt.plot(x_test, predict, color="red")
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.legend(["Actual Values", "Best Line or General formula"])
        plt.title("Scatter Regression Plot of Dataset")

    # We cannot give data directly we have to specify the values for x and y
    else:
        plt.scatter(X[:, 0], X[:, 1], cmap="viridis", c=Y, alpha=0.4)
        plt.plot(x_test, predict, color="red")
        plt.legend(["Actual Values", "Best Line or General formula"])
        plt.colorbar()
        plt.title("Scatter Regression Plot of Dataset With Target Classes")

    return fig


def data_model_description(algorithm, algorithm_type, data_name, data, X, Y):

    # Calling function to print Dataset Information
    info(data_name, algorithm, algorithm_type, data, X, Y)

    
    if (algorithm_type == "Regressor") and (
        algorithm == "Decision Tree"
        or algorithm == "Random Forest"
        or algorithm_type == "Linear Regression"
    ):
        params = add_parameter_regressor(algorithm)
    else:
        params = add_parameter_classifier_general(algorithm)

    # Now selecting classifier or regressor
    # Calling Function based on regressor and classifier
    if algorithm_type == "Regressor":
        algo_model = model_regressor(algorithm, params)
    else:
        algo_model = model_classifier(algorithm, params)

   
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)

    # Training algorithm
    algo_model.fit(x_train, y_train)

    # Plotting
    fig = plt.figure()

    # Now we will find the predicted values
    predict = algo_model.predict(x_test)

    X = pca_plot(data_name, X)

    if algorithm_type == "Regressor":
        fig = choice_regressor(X, x_test, predict, data, data_name, Y, fig)
    else:
        # Calling Function
        fig = choice_classifier(data, data_name, X, Y)

    if data_name != "Salary" and data_name != "Naive Bayes Classification":
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")

    # Since we have done pca in naive bayes classification data for plotting regression plot
    if data_name == "Naive Bayes Classification" and algorithm_type == "Regressor":
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")

    st.pyplot(fig)

    # Finding Accuracy
    # Evaluating/Testing the model
    if algorithm != "Linear Regression" and algorithm_type != "Regressor":
        # For all algorithm we will find accuracy
        st.write("Training Accuracy is:", algo_model.score(x_train, y_train) * 100)
        st.write("Testing Accuracy is:", accuracy_score(y_test, predict) * 100)
    else:
        # Checking for Error
        # Error is less as accuracy is more
        # For linear regression we will find error
        st.write("Mean Squared error is:", mean_squared_error(y_test, predict))
        st.write("Mean Absolute error is:", mean_absolute_error(y_test, predict))


# Doing PCA(Principal Component Analysis) on the dataset and then plotting it
def pca_plot(data_name, X):

    # Plotting Dataset
    # Since there are many dimensions, first we will do Principle Component analysis to do dimension reduction and then plot
    pca = PCA(2)

    # Salary and Naive bayes classification data does not need pca
    if data_name != "Salary":
        X = pca.fit_transform(X)

    return X


# Main Function
def main():

    # Giving Title
    st.title("HyperTuneML Platform")

    # Giving Title
    st.write("### ML Algorithms on Inbuilt and Kaggle Datasets")

    # Now we are making a select box for dataset
    data_name = st.sidebar.selectbox(
        "Select Dataset",
        (
            "Iris",
            "Breast Cancer",
            "Wine",
            "Diabetes",
            "Digits",
            "Salary",
            "Naive Bayes Classification",
            "Car Evaluation",
            "Heart Disease Classification",
            "Titanic",
        ),
    )

    algorithm = st.sidebar.selectbox(
        "Select Supervised Learning Algorithm",
        (
            "KNN",
            "SVM",
            "Decision Tree",
            "Naive Bayes",
            "Random Forest",
            "Linear Regression",
            "Logistic Regression",
        ),
    )

    # The Next is selecting regressor or classifier
    # We will display this in the sidebar
    if (
        algorithm != "Linear Regression"
        and algorithm != "Logistic Regression"
        and algorithm != "Naive Bayes"
    ):
        algorithm_type = st.sidebar.selectbox(
            "Select Algorithm Type", ("Classifier", "Regressor")
        )
    else:
        st.sidebar.write(
            f"In {algorithm} Classifier and Regressor dosen't exist separately"
        )
        if algorithm == "Linear Regression":
            algorithm_type = "Regressor"
            st.sidebar.write("{} only does Regression".format(algorithm))
        else:
            algorithm_type = "Classifier"
            st.sidebar.write(f"{algorithm} only does Classification")

    # Now we need to call function to load the dataset
    data = load_dataset(data_name)

    # Calling Function to get Input and Output
    X, Y = Input_output(data, data_name)

    data_model_description(algorithm, algorithm_type, data_name, data, X, Y)


# Function to include background image and opacity
def display_background_image(url, opacity):
    """
    Displays a background image with a specified opacity on the web app using CSS.

    Args:
    - url (str): URL of the background image.
    - opacity (float): Opacity level of the background image.
    """
    # Set background image using HTML and CSS
    st.markdown(
        f"""
        <style>
            body {{
                background: url('{url}') no-repeat center center fixed;
                background-size: cover;
                opacity: {opacity};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# Starting Execution of the Program
if __name__ == "__main__":


    st.set_page_config(page_title="HyperTuneML Platform")

    # Call function to display the background image with opacity
    display_background_image(
        "https://tse1.mm.bing.net/th?id=OIP.P0T0p6HZFeQh0vSt-HAR9wHaEK&pid=Api&P=0&h=180",
        0.8,
    )

    # Calling Main Function
    main()
