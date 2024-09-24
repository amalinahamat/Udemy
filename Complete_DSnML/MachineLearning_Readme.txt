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

            ii. logistic regression
            - create the graph in the form of sigmoid function
            - logistic regression predicts the output of a categorical depndent variable. therefore the outcome must be a categorical or discrete value.
            it can be either yes or no, 0 or 1, true or false, etc. but instead of giving the exact value as 0 and 1, it gives the probabilistic values which lie between 0 and 1
            - logistic regression is used for solving the classification problem
            - s - shaped curve

            types of logistic regression
            - binomial : there can be only 2 possible types of the dependent variables such as 0 or 1, pass or fail, etc
            - multinomial : there can be 3 or more possible unordered types of the dependent variable such as 'cat','dog', 'sheep'.
            - ordinal : there can be 3 or more possible ordered types of dependent variables such as 'low', 'medium' or 'high'

            sigmoid function 
            - y = 1/1+e^-(b0 + b1x)
            - bo is y-intercept
            - b1 is slope


        b. classification algorithms are used to predict discrete categories such as classifying emails as spam or non-spam

        c. SUPPORT VECTOR MACHINE (SVM)
            - powerful supervised learning algorithm used for both classfification and regression tasks
            - it finds the optimal hyperplane that separates different classes with the maximum margin, making it highly effective
            at handling complex, high-dimensional data
            - support vector is the closes datapoint to the margin
            - however, primarily it is used for the classification problem in machine learning

            - linear SVM : linear svm is used for linearly separable data, which means if a dataset can be classified into two classes
            by using a single straight line, then such data is termed as linearly separable data, and classifier is used called linear svm classified

            - non linear SVM : nonlinear svm is used for non-linealy separated data, which means if a datset cannot be classified by using a straight line,
            then such data is termed as nonlinear data and classifier used is called non-liner classifier
            - for nonlinear svm, we have to change the dimension of planes (1d converted to 2d and etc)
            - if planes consists of X and Y, then we need add new plane Z in Non linear SVM

            TYPES OF MARGIN
            - hard margin - the maximum margin hyperplane or the hard margin hyperplane is a hyperplane that properly separates the data points of different categories
            without any misclassifications. hard to predict the output
            - soft margin - when the data is not perfectly separable or contains outliers, svm permits a soft margin technique.
            each data point has a slack variable introduces by the soft margin svm formulation, which soften
            the strict margin requirement and permits certain misclassification or violations. easy to predict the output

            APPLICATIONS OF SVM
            - image recognition : classifying images into different categories
            - text classification : identifying sentiment, or spam in text data
            - bioinformatics : predicting protein structures and gene functions
            - finance : detecting fraud and making trading decisions
            - medical diagnosis : classifying disease and predicting treatment outcome

        d. k - nearest neighbour (knn) algorithm
            - knn is a powerfull supervised learning algorithm used for classification and regression tasks in machine learning
            - it works by identifying the K closest data points to a new input and making a prediction based on their labels or values

            - example : suppose we have an image of a creature that looks similar to cat and dog, but we want to know either it is cat or dog.
            so for this identification, we can use the knn algorithm, as it works on similarity measure. our knn model will find the similar
            features of the new dataset to the cats and dogs and based on the most similar features, it will put it in either cat or dog category.

            working step:
            1. select the number k of the neighbours
            2. calculate the euclidean distance of k number of neighbours
            3. take the k neares neighbour as per the calculated euclidean distance
            4. in the classification problem, the class labels of are determined by performing majority voting. the class with the most
            occurences among the neighbour becomes the predicted class for the target point
            in the regression proble, the class label is caluclated by taking average of the target values of k neares neigbours. the calculated 
            average value becomes the predicted output for the target data point

            choosing the optimal number of neighbours (K)
            - selecting the optimal value for the number of neighbours, K is crucial for the performance of the KNN algorithm
            - a small K can lead to overfitting, while larger K can result in underfitting
            - the ideal K strikes a balance between capturing local patterns and generalizing well to new data
            - technique like cross-validation and grid search can help determeine the best K value for a given dataset and problem. the choice of k
            also depend on factors like the size and complexity of the dataset, the level of noise, and the desired trade-off between accuracy and interpretability

            distance metrics in KNN
            1. euclidean metrics in KNN
                - the cartesian distance between the two points which are in the plane/hyperplane

            2. manhattan distance
                - we interested in the total distance traveled by the object instead of the displacement

            advantages and disadvantages of KNN
            - simplicity : knn is easy to understand and implement, making it a popular choice for beginners in ml
            - no assumptions : knn does not make any assumptions about the underlying data distribution, making it a robust algorithm for diverse datasets
            - versatility : knn can be used for both classification and regression tasks, providing flexibility in its application
            - sensitivity to irrelevant features : knn can be sensitive to irrelevant features, which can negatively impact its performance
            - computational complexity : knn can be computationally expensive, especially for large datasets, as it requires calculating distances between all data points
            - curse of dimensionality : knn perfromance can degrade as the number of feature increases, a phenomenon known as the "curse of dimensionality"

        e. naive bayes algorithm
        - supervised learning algorithm which is based on bayes theorem and used for solving classification problems
        - it is a probabilistic classifier which means it predict on the basis of the probability of an object
        - some popular examples of naive bayes algorithm are spam filtration, sentimental analysis and classifying articles
            a. mathematical formulation 
                P(A|B) = P(B|A)P(A) / P(B)
                P(A|B) -> Posterior probability
                P(B|A) -> Likehood probability
                P(A) -> Prior Probability
                P(B) -> Marginal Probability
            
            b. working of naive bayes classifier
                - working on Naive Bayes calssifier can be understood with the help of the below example:
                    - suppose we have a dataset of weather conditions and corresponding target variable "Play". So, using this dataset we need to decide whether we should play
                      or not on a particular day according to the weather conditions. so to solve this problem, we need to follow the below steps:
                        1. convert the given dataset into frequency tables
                        2. generate likehood table by finding the probability of given features
                        3. now, use bayes theorem to calculate the posterior probability
                        4. problem: if the weather is sunny, then the player should play or not?
                        5. solution: to solve this , first consider the dataset given

                          -  |  outlook    |    play
                          0     Rainy           yes
                          1     Sunny           yes
                          2     Overcast        yes
                          3     Overcast        yes   
                          4     Sunny           No 
                          5     Rainy           yes
                          6     Sunny           yes
                          7     Overcast        yes
                          8     Rainy           No
                          9     Sunny           No
                          10    Sunny           yes
                          11    Rainy           No
                          12    Overcast        yes
                          13    Overcast        yes

                          Frequency table for the weather conditions

                          weather       yes     no
                          Overcast      5       0
                          Rainy         2       2
                          Sunny         3       2
                          total         10      4

                          Likehood table weather conditions

                          weather       No            yes     
                          Overcast      0             5           5/14 = 0.35
                          Rainy         2             2           4/14 = 0.29
                          Sunny         2             3           5/14 = 0.35
                          all           4/14 = 0.29   10/14 = 0/71

                          applying bayes theorem

                          P(yes|sunny) = P(sunny|yes) * p(yes) / P(sunny)
                          p(sunny|yes) = 3/10 = 0.3
                          p(sunny) = 0.35
                          p(yes) = 0.71
                          so P(yes|sunny) = 0.3 * 0.71 / 0.35 => 0.60

                          p(no|sunny) = p(sunny|no)*p(no) / p(sunny)
                          p(sunny|no) = 2/4 = 0.5
                          p(no) = 0.29
                          p(sunny) = 0.35
                          so p(no|sunny) = 0.5 * 0.29 / 0.35 => 0.41

                          so as we can see from the above calculation that p(yes|sunny) > p(no|sunny)
                          hence on a sunny day, player can play the game

        f. Decision Trees
            - a supervised learning technique that can be used both for classification and regression problems,
            but mostly it is preferred for solving classification problems
            - in decision tree, there are two nodes, which are the decision node / root node and leaf node
            - it is a graphical representation for getting all the possible solutions to a problems/decisions based
            on given conditions
            - subnodes can be decision nodes or leaf node
            - output value is whether yes or no

            how does a tree decision working
                1. begin the tree with the root node, says S which contains the complete datasets
                2. find the best attribute in the dataset using attribute selection measure(ASM)
                3. divide the S into subsets that contains possible values for the best attributes
                4. generate the decision tree node, which contains the best attribute
                4. recursively make new decision trees using hte subsets of the dataset created in 3. 
                continue this process until a stage is reached where you cannot further classify the nodes and called the 
                final nodes as leaf node

            example :
            suppose there is a candidate who has a job offer and wants to decide wheteher he should accept the offer or not.
            so to solve this problem, the decision tree starts with the root node (salary attribute by ASM). The root node splits
            further into the next decision (distance from the office) and one leaf node based on the corresponding labels. the next 
            decision node furhter gets split into one decision node (cab facility) and one elaf node. finally the decision node
            splits into two leaf nodes (accepted offers and declined offer).

            salary is between $50000-$80000  -> no  -> declined offer
                                             -> yes -> office near home -> no   -> declined offer
                                                                        -> yes  -> provides cab facility  -> no   -> declined offer
                                                                                                          -> yes  -> accepted offer

            attribute selection measures
            - the key to building an effective decision tree is choosing the right splitting criteria. this determines which feature
            to use to split the data at each node. common slpitting criteria include:
                1. information gain : selects the feature that maximizes the reduction in entropy or uncertainty
                2. gini impurity : chooses the feature that minimizes the weight average of the squared probabilities of each class

            entropy in decision tree classifier
            - is the measure of uncertainty (cannot know the answer either yes or no)of a random variable, it characterizes the impurity of an arbitrary collection of examples
            - the higher the entropy more the information content

            entropy(s) =  -P(yes) log2 P(yes) - P(no) log2 P(no)
            where
            s = total number of samples
            p(yes) = probability yes
            p(no) = probability no

            example :
            for the set X = {a,a,a,b,b,b,b,b}
            total instances = 8
            instances b = 5
            instances a = 3

            entropy(h(x)) = [(3/8) log2 3/8 + 5/8 log2 5/8]
                          = - [0.375(-1.415) + 0.625(-0.678)]
                          = - (-0.53 - 0.424)
                          = 0.954

            information gain formula 

            gini index formula

            advantages of decision tree 
            1. interpretability
                users undertstand the reasoning behind the model's predictions, making them more transparent than othe algorithm
            2. flexibility
                can handle both numerical and categorical data , can be used both dor classification and regression tasks
            3. robustness
                can handle missing data, making them reliable choice for reald atasets that may contain noise or incomplete information
            4. nonlinearity
                can capture complex, nonlinear relationship in the data, allow to model complex pattern                                        

        g. Random forest algorithm
        - a popular ml algorithm that belongs to the supervised machine learning
        - can be used both for classification and regression
        - based on concept of esemble learning which is a process of combining multiple classifiers to solve a complex problem and to 
        improve the performance of the model
        - random forest is a classifier that contains a number of decision trees on various subset of the given dataset and takes the average
        to improve the predictive accuracy of the datasets and reduce overfitting

            esemble learning technique
            1. bagging
                - creates multiple decision trees from random subsets of the training data and aggregates their prediction
            2. boosting
                - sequentially trains weak models with each new model focusing on the errors of the previous one
            3. stacking
                - combines the outputs of multiple models using a meta-model that learns how to best integrate them

            for classifier - yes or no
            for regression - find mean and average

            working
            1. select random K data points from the training sets
            2. build the decision trees associated with the selected data points (subsets)
            3. choose the number N for decision trees that you want to building
            4. repeat step 1 and 2
            5. for new data points, find the predictions of each decision tree, and assign the new datapoints to the category that
            wins the majority votes

            example :
            suppose there is a dataset that contains multiple fruit images. so this dataset is given to the random forest classifier. the dataset
            is divided into subsets and given to each decision tree. during the training phase, each decision tree produces a prediction result,
            and when a new data point occurs, then based on the majority of results, the random forest classifier predict the final decision.
            
            instance -> tree -1 -> class A  -|
                     -> tree -2 -> class A  --> majority voting -> final class -> class A
                     -> tree -3 -> class b  -|

            advantages
            1. improve accuracy
                combines multiple decision trees, reducing the risk of overfitting and imporoving the overall predictive performance compraed to
                a single decision tree
            2. robustness to outliers
                the esemble nature of random forest makes it less sensitive to outliers in the training data, increasing the model's reliability
            3. handles diverse data
                can effectively handle a wide range of data types, including numerical, categorical, and mixes features, making it a versatile algorithm.
            4. automatic feature selection
                can automatically determine the most important featured, simplfying the feature engineering process for the user
                

    2. Unsupervised learning
    - Unsupervised learning deals with unlabeled data, aiming to uncover hidden patterns or intrinsic structures
    within the dataset

        a. clustering
        - clustering algorithms group similar data points together based on certain features or characteristics

        b. dimensionality reduction
        - these algorithms reduce the number of input variables while retaining important information

        a. K-means clustering
        - a popular unsupervised machine learning algorithm used to group similar data points into distinct clusters
        - k defines the number of pre-defined clusters that need to be created in the process, as if k = 2 , there will be two clusters, and for k = 3,
        there will be 3 clusters and so on

            k-means clustering is a fundamental unsupervised machine learning algorithm used to group similar data points into distinct clusters. the algorithm aims to 
            partition the data into k clusters, where each observation belong to the cluster with the nearest mean, serving as representative prototype of that cluster.

            key terminologies
            - centroid : the mean or average point within a cluster, which represents the center or core of that cluster
                    - the centroid is the mean or average point within a cluster, representing the center or core of that cluster
                    - calculated by taking the average of all data points assigned to that particular cluster
                    - important as they define the prototypes for each cluster and guide the assignement of data points to the closest cluster
                    - often visualizes as geometric shape, such as circles or stars, within the cluster of data points to highlight their central position.
            - cluster : a group of data points that are more similar to each other than to data points in other clusters.
            - elbow method : a technique used to determine the optimal number of clusters(K) by analyzing the within-cluster sum of squares(WCSS)

            STEP :
            1. select the number K (using elbow method) to decide the number of clusters
            2. select the random k points or centroids. (it can be other from the input dataset)
            3. assign each data point to their closest centroid, which will form the predefined k clusters
            4. calculate the variance and place a new centroid of each cluster
            5. repeat the third steps, which means reassign each datapoint to the new closest centroid of each cluster
            6. if any reassignment occurs, then go to step 4 else go to finish
            7 the model is ready

            elbow method step :
            1. to find the optimal value of cluster, the elbow method follow the below steps
            2. it executes the k-means clustering on a given dataset for different k values (range from 1-10)
            3. for each value of k, calculates the wcss value
            4. plots a curve between calculated WCSS (formula) values and the number of clusters k
            5. the sharp point of bend or point of the plot looks like an arm, then that point is considered as the best value of K.
            (find the first bend it will considered as the value of k)

            by using the k value we can find the centroid of each clusters, by using the centroid we can group the clusters


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




