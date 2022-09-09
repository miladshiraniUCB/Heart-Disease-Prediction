# Predicting Heart Disease

# Introduction 

Heart Diesease is one of the main causes of death in the United State and there has been several researches to understand how we can control and treat heart disease to reduce the number of death. In this work, by using the available data from [kaggle.com](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease), we try to find the features that have highest impact on heart disease by introducing a comprehensive model to predict if a person will have heart disease in future or not. 

In this work, we will use the suprevised categorial machine learning models namely: Decision Tree, Random Forest, XGBoost, and LightGBM. In addition, I will find a model with CatBoost. At the end, we will compare their results and will introduce the final model and we will introduce the features that has the highest effect on having heart disease by using the selected model.


# Data Preparation

In order to prepare the data, we will do the following:

* **Step 1:** We will convert binary `Yes` and `No` values to `1` and `0`, respectively. Similarly, in this step, we convert `Female` to `0` and `Male` to `1`. Therefore, the values that we get when we are comparing the importance of the results, are for those with values equal to `1`. for example, we can interpret of `Sex` as `female?` and the results that we find can be interpreted as how being identified as female will affect the chance of getting heart disease. 

* **Step 2:** In this step we will use `OneHotEncoder` from `SKLearn` to convert the multivariable features into numerical value. We will enforce drop first to reduce the collinearity. 

* **Step 3:** In this step we will normalize our numerical features to reduce the cost of computation. We will use `StandardScaler` from `SKLearn`. 

* **Step 4:** We will concatenate the different dataframes that we found in ateps 1-3 to get a dataframe to be used when spliting into train and test sets.

* **Step 5:** The last step is resampling by using `SMOTE` to take care of the imbalances in the data.


# Modeling

In this part, we will find a model to fit and predict our data. The metric that we will use is `recall` because we want to discover if a person will have a heart disease or not and it is important to find a model that has a higher recall score. 
In the section we will use the following methods to find a best model

1. **Decision Tree.** Our first model is *`DecisionTreeClassifier`* without tunning its hyperparameters. Then we will use *`GrdiSearchCV`* to find the best hyperparameters for our model.

2. **Random Forest.** We will use *`RandomForestClassifier`* as the second model to fit and predict the results. Initially we will not tune the hyperparameters and will fit the model, and after that we will use `RandomizedSearchCV` to find the best hyperparameters.

3. **Extreme Gradient Boosting.** We will use *`XGBClassifier`* as our last attempt to predict the data. Initially I woudl use this model without tunning the hyperparameters and after that we will use `RandomizedSearchCV` to find the best hyperparameters.

At the end, we will use the following approaches which are not part of the curriculum, but I would like to compare their results with the other approaches. These approaches are 

1. **LightGMB.** We will use *`LightGBMClassifier`* from *`LightGBM`* and we will compare its results with previous models.

2. **CatBoost.** We will use *`CatBoostClassifier`* from *`CatBoost`* and we will compare its results with previous models. The advantages of this method is that we do not need to use `OneHotEncoder` to convert the categorical features into numerical value. However, this brings us an issue which is we cannot use `SMOTE` to deal with the imbalances in the data.


# Final Model

In this section, we will introduce our final model. Since we are trying to predict if a person would have heart disease by using different features, it is rational to pick recall scores over other scores. Therefore, we will pick the model that has the highest recall score. We listed all the results in the following table

![./Images/results.png]

From this table we realize that `LightGBM` has the highest recall score and as a result we will consider this as our final model. 

According to this model the gender (female) has the highest effect on having heart disease and after that, people older than 80 years old have the highest probability of getting heart disease. Then, stroke and again age categories of 75-79 and 70-74 have the highest probability.

If we check the important features in the model derived by `CatBoost`, we see that age category, genetich health and sex (female) are the first three important features. 



# Summary and Conclusion


Since heart disease is one of the main causes of death in United States and the rest of the world, we decided to study and analyze the data on heart disease and understant how different features affect it. The data we used is from [kaggle.com](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease) that contains 319795 rows and 18 columns. Our data contains several categorical features as well as numerical features. 

In this work, we used different binary classification machine learning approaches namely, Decision Tree, Random Forest, XGBoost, LightGBM and CatBoost. To prepare our data, we converted categorical features to numerical values so that we can use them in our models. Moreover, because our data is imbalanced, we need to resample it by using SMOTE. Moreover, we splited our data to train and test sets, and we used these sets to train the model and test it, respectively. 

Since we want to have a model that can predict as much cases with heart disease as possible, we choose recall score as the maetric and the model with the highest recall is chosen. According to this metric, we realized that `LightGBM` has the highest recall number, according to which male and being older than 80 years old are the two first important features. 

It is important to note that one could get another result if they use other approaches with different tuned hyperparameters; therefore, the results of this study are not abolute and are relative. In addition, for getting a better model we would recommend gathering other data with different features and add them to the data that we used. This way, the resuls might be more reliable and the model might have better results. 





