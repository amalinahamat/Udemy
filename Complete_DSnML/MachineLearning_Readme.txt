INTRODUCTION TO MACHINE LEARNING

    - machine learning is a branch of artificial intelligence that allows machines to learn from data

TYPES OF MACHINE LEARNING ALGORITHMS

    1. Supervised learning
    - Supervised learning involves training a model on a labeled dataset. The model learn to make predictions
    by mapping input data to the output labels

        a. regression
        - regression algorithms predict continuous values, such as predicting house prices based on input features

            i. Linear regression 
            - linear regression is a type of supervised machine learning algorithm that computes the linear relationship between
            the dependent variable and one or more independent features by fitting a linear equation to observed data.
            when there is only one independent feature, it is known as simple linear regression and when there are more than one feature,
            it is known as multiple linear regression
            line salary and works

                a. simple linear regression
                - if a single independent variable is used to predict the value of a numerical dependent variable, then such a 
                Linear regression algorithm is called Simple Linear Regression

                b. multiple linear regression
                - if more than one independent variable is used to predict the value of a numerical dependent variable, then such a
                Linear Regression algorithm is called Multiple Linear Regression

            slope and intercept
            y = wx + b
            y - dependent variable
            x - independent variable
            w - slope
            b - intercept
            
            find best fit line:
            - when working with linear regression, our main goal is to find the best fit line that means the error between predicted values
            and actual values should be minimized. the best fit line will have the least error

            for linear regression, we use mean square error (mse) cost function, which is the average of squared error occured between the predicted values 
            and actual values. it can be written as:

            MSE = 1 N ∑ i = 1 N y i − y p 2
            for the above linear equation mse can be calculated as:

            n = total number of observation
            yi = actual value
            a1xi + a0 = predicted value 

            find model performance (r-squared)
            - r-square is a statistical method that determines the goodness of fit.
            - it measures the strength of the relationship between the dependent and independent variables on a scale of 0=100%
            - the high value of r-square deetrmined the less difference between the predicted value and actual value and hence represent the good model
            - it is also called a coefficient of determination, or coefficient of multiple determination for a multiple regression
            - it can be calculated from the below formula
            - r-square = explained variation /  total variation
            - r-square values range from 0 to 1 with 1 indicating that the regression model perfectly fits the data and 0 indicating no linear relationship 
            between the variables
            - high r - high performance

            disadvantages of linear regression
            - limited to linear relationship - may not capture complex, nonlinear patterns in the data
            - sensitive to outliers - extreme data points can greatly impact the model's fit and predictions
            - assumptions can be violated - regression requires assumptions like normality, homoscedasticity and independence to be met, which
            may not always hold True

        b. classification algorithms are used to predict discrete categories such as classifying emails as spam or non-spam


    2. Unsupervised learning
    - Unsupervised learning deals with unlabeled data, aiming to uncover hidden patterns or intrisic structures
    within the dataset

        a. clustering
        - clustering algorithms group similar data points together based on certain features or characteristics

        b. dimensionality reduction
        - these algorithms reduce the number of input variables while retaining important information


APPLICATIONS OF MACHINE LEARNING

    1. healthcare
    - machine learning is used in disease diagnosis, personalized treatment and drug discovery

    2. e-commerce
    - it is employed for product recommendations, fraud detection and dynamic pricing

    3. finance
    - machine learning helps in risk assessment, algorithms trading and fraud prevention


JOB OPPORTUNITIES
    1. python
    2. data science
    3. ai engineering




