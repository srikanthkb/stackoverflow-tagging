# stackoverflow-tagging
 Tagging of stackoverflow questions

#### Problem Statement
Extract the appropriate tags from the question and description in stackoverflow

#### Objectives
1. Predict system with high precision and high recall for as many tags as possible.
2. Incorrect tags to be reduced, as it affects customer experience.

#### Constraints
No time/latency constraints are assumed for this project.

#### Data

Dataset obtained from the Kaggle competition: https://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction/data.

Here's few insights about the data:
- train.csv: contains 4 columns - Id, Title, Body, Tags
    - Number of rows = 6034195
    - Uncompressed .csv size is 6.75GB
- test.csv: contains 3 columns - Id, Title, Body
    - Uncompressed .csv size is 2GB


#### Thoughts about the problem
Consider a question on stackoverflow, it can have multiple tags. Also, we need to note that the tags for this are not
exquisite for the question, that is, there could be two questions which can be sharing a few tags between them.
Each question can have multiple labels assigned to it. This is a "multi-label" classification problem.

#### Performance metrics
Per the objectives, we require a high precision and high recall for the model. Also from the competition evaluation rules,
we know that the MeanFScore is the metric used for measuring model efficiency. For this reason,
we use F-1 score as the performance metric, because F-1 score is the geometric mean of precision and recall.
It is only high iff both precision and recall are high.

But F-1 score is primarily defined for a binary classification problem. Using the metrics presented in the evaluation
we use the MeanFScore, which is also called as micro-averaged F1-score. In this case, the F1-score is the
harmonic mean of micro-precision and micro-recall (to adapt the F1 score to a multilabel situation). The definition of
micro-precision and micro-recall is defined in the evaluation section of competition.

Micro-precision and micro-recall have an altered formula, which pays higher weightage to more contributing class labels.
This is done, to maintain the ratio for F1-micro.

Micro-averaged F1 scores are particularly useful in the cases of unbalanced datasets.

Hamming loss measures the accuracy in a multi-label classification task. XOR logic is used for measuring the hamming loss

### Exploratory Data Analysis

#### Loading the data
Load the Train.csv files using Sqlite and pandas.

Understanding the distribution of frequency of different tags (label)

![image](https://user-images.githubusercontent.com/36214903/148880422-56f653ce-642d-4723-8093-2dc622ed4b13.png)

Number of tags in different questions

![image](https://user-images.githubusercontent.com/36214903/148880461-7c509f33-c53b-48d5-ab0e-7fc80a43dc54.png)

##### Observations
1. Maximum number of tags per question is 5
2. Minimum number of tags for a question is 1
3. Average number of tags for a question is 2 or 3.


##### Word frequency cloud
One of the best ways to visualize the most frequent tags is to use a word cloud. The size of the word
in this image/plot represents the frequency of the word/tag.

![image](https://user-images.githubusercontent.com/36214903/148880480-ced49ef0-4812-4d90-96f0-cd9d7a4d46f4.png)

Visualize top 20 tags

![image](https://user-images.githubusercontent.com/36214903/148880510-b1c8cd23-443b-488e-bd1a-d1e637951e47.png)

Here's few insights from the above bar plot
1. Most commonly, the programming languages are represented as tags
2. Android, ios, linux and windows are frequent tags related to operating systems
3. sql is the most common tag related to databases.

### Data Preprocessing

Here's the steps taken into data preprocessing:
1. Reduce the train sample from 4M to 1M to accommodate for computation limitations. We instead randomly sample 4M
data points to obtain 1M data points.
2. Seperate and remove code-snippets from question body (as a part of text preprocessing)
3. Remove special characters (?) from question title and description (except for code - because they represent keywords/
operators)
4. Remove the stop words (except for 'C'), because they carry very little or no useful information.
5. Remove HTML tags, because the question is written down in markdown languages, we have no use for html tags (<, >).
6. Convert all characters to small case (for uniformity)
7. Stem the words, to their root words (example: definition, defining, define are replaced by defin)

### Machine learning and Modelling

#### Notes on multi-label classification problems
One way to solve multi-label classification problem, is to use binary relevance. In this method, the classification problem
is broken down to different problems for each class label, i.e. in our case, it will lead to 42k different binary
classification problems. Works well for a small dataset with few labels to classify.

Second method, is to use classifier chain method. In which, we divide the problem to classify for class 1,
then we use the outputs in class 1 and X (inputs) to predict the labels for class 2. This is further repeated for all classes.
In this case, we still end up solving at least n classification problems (same as binary classification). This is useful,
if the class labels are correlated among themselves.

Third method, is to use the label powerset. In this method, the predicted labels are grouped to represent a single label
(in integer), and then, any other inputs which have these same labels are represented by the new label for the set of labels.
This method might not scale well with the number of labels to predict, as different labels combination will yield a different
class output for the label powerset. Ideally, if this were a binary vector, then the number of powerset classifiers can be
upto 2^(number of labels)

Fourth method, is the adapter algorithm from Multi-label KNN approach.

Ref: https://www.analyticsvidhya.com/blog/2017/08/introduction-to-multi-label-classification/


As we noticed in the frequency plot for the labels had high skew, we know that all 42k tags are not equally significant.
This gives rise to two questions:
- How many tags are to be chosen ?
- What percentage of questions are covered with the number of tags that we choose ?

![image](https://user-images.githubusercontent.com/36214903/148880632-22b91ee8-287a-4ded-abc9-f03b27cab5ab.png)

Here's few insights from the above plot:
- Choosing 500 questions will cover upto 90% of the questions
- Choosing 5500 questions will cover upto 99% of the questions
- 18k tags will cover nearly 100% of the questions


### Results
Using a OneVsRest SGD Classifier from Sklearn, the label prediction works with an average precision of 0.72, average recall of 0.33 and an F1-score of 0.45

Low recall seems to be the factor in bringing down the F1-score.

These scores can be further improved with techniques such as:
- Using 1M data points (instead of 0.5M) 
- Using 5000 data labels (instead of 500 labels)
- Using stacking techniques (which takes more time to train different classifiers)
- Adding more weights to titles and code part of the question.
