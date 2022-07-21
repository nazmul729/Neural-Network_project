# Performance Evaluation of Neural-Tree based Ensemble method on Twitter sentiment analysis
The use of Social media are increasing day by day to express individual feelings and opinions in the form of text. Millions of user share their feelings and opinions (Emotion) on different aspects of everyday life. To detect these emotions, Machine-based emotional intelligence is required for more natural interaction between humans and computer interfaces. In recent years, lots of research works have been done to detect emotions from text and/or social media. In this investigation, a survey cum
experimental methodology was adopted to give focus on Twitter data as twitter is one of the most popular social media and micro blogging platform. This research aimed to improve performance of textual emotion in two ways. 
- First, extracted some features and then compared performance among several data mining and machine learning algorithms on textual emotional data. 
- Second, the best classifiers were selected from the comparison and used them to make ensemble classification using voting method.

Among them, our voting method have shown better performance than the invidividual methods.

## Dataset and preprocessing

In this project, I have collected tweets using the Twitter API. I intentionally chose to use some overlapping topics between tweets in order to encourage cross-language approaches. I filtered the tweets for duplicates. The training and test sets were created from the 2017 annotations. Thus, the scores for tweets across the training and
and test sets are not directly comparable. However, the scores in each dataset indicate relative positions of the tweets in that dataset. [The Training Set was taken from EmoInt 2017, rereleased Aug, 2017; last updated Nov. 23, 2017 and the Development Set was released Sep. 25, 2017; last updated Nov. 23, 2017 and all test tweets were collected over the period of December 2016-January 2017](https://arxiv.org/abs/1804.06658). The data files include the tweet id, the tweet, the emotion of the tweet and the emotion intensity (for training and test sets). There are around 7103 tweets in the training set, and 71817 in the test set (across all the emotions).

Several preprocessings have been used,

- **[Data Cleaning](https://ieeexplore.ieee.org/document/7050801)** Tweets are not always syntactically well-structured and sometimes there is no meaning of the tweets. All of the annotated lexica are also needed to be cleaned in the same way as the tweets are.
- **[Tokenization](https://dl.acm.org/doi/10.3115/992424.992434)** Tokenization is a process to split sentence or longer string of text into smaller pieces, or tokens. Larger chunks of text can be tokenized into sentences, sentences can be tokenized into words, etc. Tokenization is necessary due to text lexical analysis.
- **[Normalization](https://pubmed.ncbi.nlm.nih.gov/23043124/)** Normalizing text can mean performing 3 distinct jobs: 
  1. Stemming is the process of eliminating unnecessary words from a word in order to obtain a word stem. 
  2. lemmatization is the ability to capture canonical forms based on a word's lemma. 
  3. The third option contains a list of tasks: remove stop words, punctuations, web addresses, blank spaces, the numbers, and convert the entire text to lower case.

## Feature extraction
I have evaluated several feature extraction methods like,
  1. ***Total Emotion Count (TEC)***
  2. ***Graded Emotion Count (GEC)***
  3. ***N-Grams***
  4. ***Part-of-Speech (POS) features***
  5. ***Word HashTags***
  6. ***SentiWordNet (SWN)***
  7. ***Word2Vec Model***
  8. ***TF-IDF Value***
  9. ***Entropy Based***

## Proposed Neural-Tree based Architecture
![Voting method](https://user-images.githubusercontent.com/42664968/180129983-1cf7d14e-c8cd-48ae-b196-6dc762a0603d.PNG)

Model-1: Using three Tree-based algorithms-
  - Naive Bias
  - Support Vector Machine
  - Decision Tree Classifier (C4.5)
  
Model-2: Using three MLP algorithms-
  - MLP with quasi-Newton methods
  - MLP with SGD
  - MLP with SGD by Kingma

Model-3: Using 3 SGD algorithms
  - SGD with soft-margin
  - SGD smoothed hinge loss Parameter
  - SGD with logistic regression


## Empirical Evaluation

![Voting method](https://github.com/nazmul729/Neural-Network_project/blob/main/model1.png)
![Voting method](https://github.com/nazmul729/Neural-Network_project/blob/main/model2.png)
![Voting method](https://github.com/nazmul729/Neural-Network_project/blob/main/model3.png)
![Voting method](https://github.com/nazmul729/Neural-Network_project/blob/main/allmodel.png)


## Credits

- Special thanks to [Professor Dr Anthony S. Maida](https://people.cmix.louisiana.edu/maida/)
- Implemented this project using Python frameworks: Numpy, Pandas, Scikit-learn.
