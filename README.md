# Natural Lnagugage processing models 


**Abstract**

In this project, two models were used for text mining: Multinomial Naive Bayes and BERT. Both models attained high accuracy, but BERT outperformed Naive Bayes with a 93% accuracy compared to Naive Bayes with 85% accuracy. The Naive Bayes model was optimized by performing a grid search over the alpha Laplace smoothing parameter, the maximum number of features, and the training set size. In the second phase of the project, for the BERT model, a pre-trained BERT model was implemented with an 80/20 train-test split. A grid search was conducted to find the optimal hyperparameters and it was found that the highest accuracy, 93%, was achieved with a learning rate of 2e-5, a maximum token length of 512, and a batch size of 32. In the last part, multi-layer multi-headed attention was added to the BERT model to see the attention scores of each token for some correctly and incorrectly predicted sentences. The attention matrix helped to recognize the relation between tokens better.

**# 1 Introduction**



This mini-project involves implementing two machine learning algorithms: Naive Bayes from scratch and BERT using pre-trained weights. The purpose of the project is to gain practical experience in implementing machine learning algorithms for natural language processing. Specifically, we will compare the performance of these two algorithms on a real-world textual dataset, namely the IMDB review dataset. We found out that multinomial Naive Bayes with Laplace smoothing hyperparameter alpha =1 and a maximum number of features set to None reaches a significant accuracy of around 85%. Bert transformer, on the other hand, with a learning rate of 2e-5, a maximum token length of 512, and a batch size of 32, has achieved an accuracy of 93%. This dataset contains a large collection of movie reviews that were scraped from the IMDb website. It has been used in several studies, which entails categorizing text as positive or negative. For instance, in [1], they proposed a method for learning word vectors for sentiment analysis using this dataset. They showed that their method outperformed traditional bag-of-words models. Another paper, [2], conducted a comparative study of pre-trained transformers for aspect-based sentiment analysis using the IMDB dataset. Finally, in [3], the authors fine-tuned the pre-trained BERT model on the IMDB dataset and compared its performance to other state-of-the-art models for sentiment analysis.


**# 2 Dataset**


The IMDB dataset comprises an extensive collection of movie reviews from the IMDb website, containing almost 50,000 reviews with equal proportions of positive and negative feedback. This dataset has been widely utilized for sentiment analysis and other natural language processing tasks. Common preprocessing steps for the IMDB dataset include tokenization, lowercase conversion, stopword removal, tags removal, and stemming or lemmatization. Tokenization involves breaking down the text into individual words or tokens, followed by converting them to lowercase for standardization. Stopword removal eliminates common words that do not contribute much to the meaning of the text. Stemming or lemmatization reduces words to their root forms for data dimensionality reduction and capturing core word meanings.
