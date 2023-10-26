# NLP_Sentiment_Analysis

Sentiment analysis using Twitter data concerning vaccines (neutral, anti-vax and pro-vax) for a university course. The datasets are CSV files with the tweet texts and the labels 0 (neutral), 1 (anti-vax) or 2 (pro-vax). The first dataset is the training dataset and the second one is the validation dataset. The model was evaluated by the course's instructors on a test set that was not be provided, but was of the same format as the training and validation datasets. 
* **The training and validation datasets must be uploaded at your google drive.**

Sentiment analysis (classification) is done using 4 different techniques.
* Softmax Regression
* Feed Forward Neural Networks (FNNs)
* Recurrent Neural Networks (RNNs) with LSTM/GRU cells
* BERT-base model by [Huggingface](https://huggingface.co/bert-base-uncased)

## Development Environment and Libraries

* Google Colab was used for training and fine-tuning of the 
models.
* Python > 3.6
* PyTorch 
* MatPlotLib & Scikit Learn Libraries for implementing the model and plotting graphs
* NLTK for loading and preprocessing the data

## Softmax Regression

### WORKFLOW OF THE PROJECT

* The first step is to load our data into our project. So, the two given data sets are loaded; the vaccine_train_set, which will be used for the training of the classifier and the vaccine_validation_set, which will be used for evaluating the classifier. 

* The second step is to preprocess these data, before they are used by the classifier. We clean the data by removing the “@example” annotations, the “http://example.com” links and every sort of tokens, such as #, (, ), -, _, :, ;, etc. Furthermore, we implemented Lemmatization and Stemming. Both lemmatization and stemming are processes through which similar words are grouped together, so that they can be analyzed as a single term and they are identified by the same term. For example, words like “goes”, “going”, “went” are replaced by the word “go” and the word “friendship” is transformed to “friend”. That way, a lot of noise existing in the data as well as redundant data, which is not useful for the classifier, is being cut off.

* The next step is to create features for our model. For this reason, we use classes from Scikit Learn, such as CountVectorizer, HashingVectorizer and Tfidf_Vectorizer, which convert the text from our data sets into features with numbers. Then, we provide the data from the vaccine_train_set to the model, so that it can learn the vocabulary and convert it to an array of numbers. Then, we provide the data from the vaccine_validation_set to the model in order to convert it to an array of numbers (it does not learn new vocabulary). After that we use Standard Scaler in order to standardize features by removing the mean and scaling to unit variance. It was observed that the model did not perform very well without it, because there are many, sparse data. Hence, scaling functions as a smoothing factor.

* Then, we use LogisticRegression from Scikit Learn to implement our classifier and we give it the previously transformed-to-numbers data in order to train it. Later, we provide the classifier separately with the vaccine_train_set and the validation_data_set, so that it can make predictions for the given data; which class should every instance have.

* We have implemented the model the first time using CounterVectorizer, the second HashingVectorizer and the third Tfidf_Vectorizer. We have experimented with several variants of each of the above ways and some of the results are presented below.

* Finally, we plot the Learning Curves with the train_data_set size and F1scores (for train and validation set) for the different variants. Moreover, we calculate the metrics of our model at each case (precision, recall, f1score, accuracy), so that we can evaluate how good the classifier is. At some cases, precision, recall and f1score are calculated separately for each class in order to evaluate how well or badly the model predicts a specific class.

### RESULTS – OBSERVATIONS – GENERAL COMMENTS:
* For the implementation of the Softmax Regression, Logistic Regression from Sklearn was used setting the solver to “lbfgs” and the multiclass to “multinomial”, because we have to classify the data in more than 2 classes (here 3).

* Solver “liblinear” with the multiclass being set to “ovr” was tested too, but it was a lot slower than the first option. So, it was rejected and all the following results are with respect to solver = “lbfgs” and multiclass = “multinomial”. 

* The parameter “max_df” at all the ways used to create the features of the model was set to 5000 respectively, so that the features, which appear too many times in the data set, are excluded from our model’s features’ list. Min_df was not changed, since we choose each time the N max_features; features with higher frequency among the data.

* All metrics for the evaluation of the model are calculated 2 times. The first time we set average = “macro”, in which all classes have the same weight. The second time we set average = “weighted”, which takes into account the weight of each class. This is useful, because all the classes do not have the same number of instances. Some have more than others. 

* **In all graphs, the metrics have average=”macro”.** That way, the evaluation is more objective. Otherwise, according to sklearn documentation for f1score; it may not be between precision and recall, if average = “weighted”. However, f1score with average = ”weighted”, was also calculated (as told in the previous bullet) and the results can be seen on the notebook.

* Generally, the scores for the metrics (precision, recall, f1score) are higher, when average = “weighted”, than the ones when average = “macro”. This happens, because the evaluation with macro is more objective. The fact that a class has more instances than others does not imply that it is more important than others in the evaluation in our case. 


### CHOICE OF THE MODEL -> USING THE COUNT VECTORIZER

It was observed that the model with the best performance uses the Count_Vectorizer for the transformation of the data into numbers. After trying many different values in order to make our model better, we have finally reached the conclusion that the model will have: 
*	max_features = 800. It is an average number of features considering the training_set_size: not too few, not too many.

*	C = 0.5 (a bit of regularization to our data in  order to achieve better results)

*	Class_weight = None (we have only 3 classes and for example class 0 has more instances than class 1 and class 2). After experimentation, it turned out to be better when class_weight is None rather than “balanced”, because when it is “balanced”, the model during learning considers better to make the predictions with a more balanced probability among the 3 classes. So, it predicts less for class 0 which has the most instances.

*	Data clean and lemmatization (no stemming). Stemming turned out not to be very helpful. It produced worse results for f1score for both the validation and training sets.

From now on, we will refer to this model as **best_count**.
You can find the metrics for this model for both training and validation sets, and separately (only for the validation set) the metrics for each class of the data under the *FINAL MODEL* heading in *Softmax_Regression.ipynb*.

### Experiments
*	Using Hashing_Vectorizer
It turned out while experimenting with Hashing_Vectorizer that it was the fastest of all (Count_Vectorizer, TfIdf_Vectorizer), but it had the worst performance at all experiments. F1score and accuracy were around 58-63% at all variants. The learning rates were generally lower than for the Count_Vectorizer and TfIdf_Vectorizer and when we were increasing C, the learning rates were smoother. Maybe, Hashing_Vectorizer may be ideal for larger datasets with more features, because with few features there can be collisions.

*	Using TfIdf_Vectorizer
It turned out that the performance of the model -for both the training and the validation sets- is similar with the ones using the Count_Vectorizer . When the model has learned the whole training_set, the scores for the metrics are similar with the ones the best_count has (65% for f1score - 69-70% for accuracy). F1score = 63% and accuracy = 67% for the validation set. During the training, we see from the graphs (at the experiments) that f1score increases –at first faster and then more slowly. So, the learning rate is a little worse than the one of best_count. One more remark is that this vectorizer was the slowest of all. For these 2 reasons we chose Count_Vectorizer for our model.

*	No data pre-processing
Whether we use Count_Vectorizer, Hashing_Vectorizer or Tfid_Vectorizer (when there is no data pre-processing at all), the performance of the model -for both the training and the validation sets- is equally good or at some cases a little better than the one of best_count. At some cases the scores for the metrics are 0.5% - 1% worse than the ones the best_count has, or at some others they are 0.5% - 1% better. However, during the training, we see from the graph that f1score increases until one point, then decreases and then increases again. So, the learning rate is a little worse than the one of best_count. It seems that it increases more at the end but we would need more training instances to examine that. That is the reason why we chose best_count as the best model.

*	class_weight = None
Whether we use Count_Vectorizer, Hashing_Vectorizer or Tfid_Vectorizer (when we set class_weight = None), the performance of the model -for both the training and the validation sets- becomes better than the performance when we set class_weight = “balanced”. More specifically accuracy and f1scores for both training and validation sets are about 2%-3% higher, when class_weight = None. It does better predictions for the validation set and the learning rate gets better (i.e. faster). So, when all classes the same weight gives better results, maybe because the validation set is relatively small. (2283 instances)

*	Full data pre-processing
Whether we use Count_Vectorizer, Hashing_Vectorizer or Tfid_Vectorizer (when we have Full data pre-processing), the metrics for the validation set are similar with the ones of the best_count. However, the metrics for the validation set are worse than those of the model which did not have any pre-processing at all. They are about 3%-4% worse. This may happen because a lot of noise is removed from the data and we need for instances to have more important features. F1score is around 61% and accuracy is around 64%. The behavior of f1score in respect with the training size, however, is worse for both sets.  So, learning rates are slower and less smooth, which is not very good. 

*	Lower number of max_features (e.g. 100)
Whether we use Count_Vectorizer, Hashing_Vectorizer or Tfid_Vectorizer, decreasing the number of the features a lot leads to:
    1.	Lower precision, f1score and accuracy values (at about 1%-1.5%) for the training set, than those the best_count model has. 
    2.	Lower precision, f1score and accuracy values (at about 1%-2%, or more at some cases, mainly when using HashingVectorizer) for the validation set, than those of the best_count model. This happens, because the validation set is smaller and it needs fewer features to make predictions, but not too few, because the model cannot learn sufficiently. The learning rates are faster but they are not increasing at a fixed rate.

*	Greater number of max_features (e.g. 7000)
Whether we use Count_Vectorizer, Hashing_Vectorizer or Tfid_Vectorizer, increasing the number of the features leads to:
    1.	higher precision, f1score and accuracy values for the training set than those the best_count model has. We can reach f1score and accuracy at around 90%-95%, if the number of features is very high.
    2.	similar or lower precision, f1score and accuracy values for the validation set  than those the best_count model has (e.g 800≤max_features≤1300). This happens, because the model reaches a point where no more features can make it better for the validation set, because the validation set is relatively small. So the more exist, the more they confuse it. The learning rates are similar.

    Having e.g. 7000 would mean that the model overfits our data, because the f1score and accuracy can easily get 10% lower. 

*	Whether we use Count_Vectorizer, Hashing_Vectorizer or Tfid_Vectorizer, an increase in C (inverse of regularization strength) provides less regularization for the data, which at some cases lowers f1score and accuracy for the validation set and at some others these metrics remain the same. However, the higher value C has, the more time-consuming it is for the model to be executed.

*	In the Jupiter notebook, experiments with combinations of the above are provided, which produce a combination of the above results. It depends on which feature is more powerful each time.

### CONCLUSION
Comparing the best version of the models of each category above, we have to admit that they perform equally well more or less. However, the best seemed to be the one with the count_vectorizer. Next comes the one with the tfidf_vectorizer and last the one with the hashing_vectorizer.

## Feed Forward Neural Networks (FNNs)

At the first model we use pre-trained [GloVE](https://nlp.stanford.edu/projects/glove/) word embeddings (more specifically Glove6B of different dimensions) and at the second we use Tfidf_Vectorizer.

### WORKFLOW OF THE PROJECT

*	The first step is to load our data into our project. So, the two given data sets are loaded; the vaccine_train_set, which will be used for the training of the neural network and the vaccine_validation_set, which will be used for evaluating the model. 

*	The second step is to preprocess these data, before they are inserted into the neural network. We clean the data by removing the “@example” annotations, the “” links and every sort of tokens, such as #, (, ), -, _, :, ;, etc. We do not remove stop_words, because they may be useful for our neural network since they are likely to appear many times in the data.

*	The next step is to create features for our model. For this reason, we use pre-trained Glove word embeddings provided by Stanford for the first model. We create an array, which has all the vectors for the words in the glove.txt file and we keep a dictionary having the words of the glove.txt file as keys and the index of the row of the above array as values, which corresponds to a specific word. For every sentence in our data we get the corresponding vectors for the words that exist in the glove.txt and create an array out of them. For words that do not exist we insert to the above array a row with randomly generated numbers. At last, for every feature of the model (every column), we get the mean of all the values in the rows. For the second model we use the class Tfidf_Vectorizer from Scikit Learnd for second model, which convert the text from our data sets into features with numbers. Then, we provide the data from the vaccine_train_set to the model, so that it can learn the vocabulary and convert it to an array of numbers. Then, we provide the data from the vaccine_validation_set to the model in order to convert it to an array of numbers (it does not learn new vocabulary). After that we use Standard Scaler in order to standardize features by removing the mean and scaling to unit variance. It was observed that the model did not perform very well without it, because there are many, sparse data. Hence, scaling functions as a smoothing factor.

*	Then, we define the neural network, the hyperparameters and we train the model for 100 epochs with a specific batch size. At the end of every epoch we evaluate our model providing the data of the validation_data_set, so that it can make predictions for the given data; which class every instance should have. 

*	Finally, we plot the Learning Curves with the train_data_set and the validation_data_set with respect to every epoch to see how well the validation set has learned. We plot also the Roc curve to see the dependency between the true positive ratio and the false positive ratio. Moreover, we calculate the metrics of our model at each case (precision, recall, f1score, accuracy), so that we can evaluate how good the classifier is. At some cases, precision, recall and f1score are calculated separately for each class in order to evaluate how well or badly the model predicts a specific class.

*	We have experimented with several variants of each of the above ways and some of the results are presented below.

### RESULTS – OBSERVATIONS – GENERAL COMMENTS:

*	For the implementation of the Neural network, we create a class which inherits nn.Module class and implement a constructor and a function for forward propagation. 

*	We observed that shuffling the data gives more diversity among the results (metrics, curves). They are similar to those without having shuffled the data, so we shuffle them to simulate a more real case scenario.

*	At each model the number of neurons at input layer is equal to the number of features of the model and the number of neurons the output layer is 3, because we have 3 classes.

*	All metrics for the evaluation of the model are calculated 2 times. The first time we set average = “macro”, in which all classes have the same weight. The second time we set average = “weighted”, which takes into account the weight of each class. This is useful, because all the classes do not have the same number of instances. Some have more than others. 

*	**In all graphs the metrics have average =”macro”.** That way, the evaluation is more objective. Otherwise, according to sklearn documentation for f1score; it may not be between precision and recall, if average = “weighted”. However, f1score with average = ”weighted”, was also calculated (as told in the previous bullet) and the results can be seen on the notebook .

*	Generally, the scores for the metrics (precision, recall, f1score) are higher, when average = “weighted”, than the ones when average = “macro”. This happens, because the evaluation with macro is more objective. The fact that a class has more instances than others does not imply that it is more important than others in the evaluation in our case. 

### CHOICE OF THE MODEL-> USING THE TF-IDF VECTORIZER

It was observed that we achieve the best performance for the neural network when we use the TF-IDF Vectorizer for the transformation of the data into numbers, and not the pre-trained Glove word embeddings. After trying many different values in order to make our model better, we have finally reached the conclusion that the model will have: 
*	max_features = 1000. It is an average number of features considering the training_set_size: not too few, not too many.
*	Learning_rate = 1e-4
*	One input layer, 3 hidden layers with 35, 15 and 8 neurons respectively and one output layer
*	Activation function = RELU() at each hidden layer
*	Loss_func = CrossEntropyLoss()
*	Optimizer = Adam()
*	Epochs = 100
*	Training_batch_size = 200 
*	Data preprocessing as explained above. 

You will find in the *Neural Network using TF-IDF vectorizer -> Training and Evaluation using CrossEntropyLoss* header the metrics for this model for validation set, the loss of both the training and validation sets with respect to the training epochs and the ROC curve of the true positives ratio with respect to the false positives ratio.

From the first curve we can see that the validation set learns from the training set well, and it is unlikely that there is overfitting or underfitting in the data. The lines of the training and validation loss are very close to each other at the early epochs. Both the training and the validation loss decrease as we continue the training for more epochs.

From the ROC curve we infer that the model does more correct predictions for the class 0 (the pink line is closer to the vertical axis than the cyan and the blue lines). That means that the true positives ratio is bigger than the false positives ratio from the beginning of the evaluation and that the model does many correct choices and it does not classify many wrong labels as class 0. For the class 1, its predictions are a bit worse and for the class 2 its predictions are worse than those of class 1.  At the end of the evaluation, it is obvious that only few labels are mistaken to belong to a different class.

According to the above observations, it is expected that the precision of the class 0 to be higher than the one of class 1 and of class 2. We can verify that from the results below. 

For class 0 we have precision = 0.77, for class 1 we have precision = 0.61 and finally for class 2 we have precision = 0.68. The total precision of the model is 0.68 and the accuracy is 0.72. The f1score for class 0 is 0.77 and higher than f1scores of the other classes as it was expected. The f1score for class 1 is 0.51 and for class 2 is 0.72. The total f1score of the model is 0.66. Our model makes better predictions for class0 and class2. As far as class 1 is concerned, the model predicts many instances to belong to this class, while in fact they don’t. This is obvious because of the low recall score of this class, which of course affects the f1score as well, which is also low; 0.51.

### Experiments
You can see all plots in the *Feed_Forward_Neural_Networks* notebook.

*	Using SGD optimizer: 

    We use optimizer Adam at all implementations because we observed that it worked better than SGD optimizer. We get higher values for the metrics by using Adam and better Loss and Roc curves.

*	Using MSELoss() vs using CrossEntropyLoss():

    Both Loss Functions seem to work equally fine with our data.  We achieve f1score around 57% with CrossEntropyLoss and f1score around 56% with MseLoss when using Glove word embeddings. We achieve f1score around 66% with CrossEntropyLoss and f1score around 65% with MseLoss when using Tf-Idf Vectorizer. Generally CrossEntrpoyLoss function tends to work better with unbalanced classes like those we have here than MSELoss function. At the left is MSELoss and at the right is CrossEntropyLoss using Glove word embeddings of 200 dimensions and the corresponding ROC Curves.

    We can see that the model with TF-Idf vectorizer presents better Roc Curves. That was expected since it has higher f1 scores. It can make better predictions for classes 1 and 2 than the one which uses Glove word embeddings.

*	Training at more / less epochs:

    The more epochs we have the better the model is trained, because we achieve lower values for losses and the model generally learns more.

*	Using bigger / smaller batch size:
    
    When we use smaller batch size the model learns more slowly because it does not have many samples and this has an impact on the curves and the scores for the metrics.  F1score is around 59% - 60% when using Glove. When using TF-IDF f1 score is around 64%-66%.

    On the other hand if we use a very large batch size (equal to the validation set’s size for example – batch_zise = 2000) the model is highly likely that f1score may fall (around 55% when using Glove and around 50% when using TF-IDF Vectorizer). However, from the loss functions we understand that the model is very good but the metrics’ scores are very lower, so it does not make good predictions. 

*	Using bigger learning rate:

    It turned out that the performance of the model for the validation set is worse when using learning rate bigger than 1e-4. The curves are worse but f1score remains similar as when having learning rate 1e-4. The curves below are for MSELoss when using Glove. For the other combinations are similar as well.

*	Activation function at each hidden layer vs at none:

    The activation function tested is RELU(). When we do not use any activation function at all when using Glove, we can see that the performance of the model at the validation set is worse rather than when we use some activation function (that is why we use activation function at each layer of the neuron network at our chosen model).

    *	MSELoss when using Glove and f1score = 47%
    *	CrossEntropyLoss when using Glove and f1score = 46%
    *	MSELoss when using TF-IDF Vectorizer and f1score = 63%
    *	CrossEntropyLoss when using TF-IDF Vectorizer and f1score = 65%
*	Less / More layers & Less / More number of neurons at each layer:

    Inserting at the neural network more neurons per layer (80, 35, 10) makes it more complex and we achieve similar scores as those of our chosen model for metrics and roc curves; around 57% - 58% when using Glove and around 65% - 66% when using TF-IDF vectorizer. However the loss curves seem to be worse, as there is bigger space between the lines of the training and the validation set. When we apply less number of neurons (30, 10, 7) per layer, f1scores are about 52% and 48% for mse and crossEntropy respectively.  

* In the *Feed_Forward_Neural_Network* notebook, experiments with combinations of the above are provided, which produce a combination of the above results. It depends on which feature is more powerful each time. 

### CONCLUSION
Comparing the best version of the models of each category above, we have to admit that they perform equally well more or less. It depends both on the data sets and on the choice of the technic and the selection of the hyperparameters.

## Recurrent Neural Networks (RNNs) with LSTM/GRU cells

### WORKFLOW OF THE PROJECT

*	The first step is to load our data into our project. So, the two given data sets are loaded; the vaccine_train_set, which will be used for the training of the neural network and the vaccine_validation_set, which will be used for evaluating the model. 

*	The second step is to preprocess these data, before they are inserted into the neural network. We clean the data by removing the “@example” annotations, the “” links and every sort of tokens, such as #, (, ), -, _, :, ;, etc. We also remove stop_words, because they may not be useful for our neural network since they are likely to appear many times in the data.

*	The next step is to create features for our model. For this reason, we use pre-trained Glove word embeddings provided by Stanford (here we chose the vectors with 300 dimensions) and we will use them to create the first embedding layer of our models. So, we find words of our training set into this glove6B200.txt file and we replace each word of the training set with the index of the row of the glove file in order to connect this word with the pre-trained vector. For the words of the training set, which do not correspond to any of the vectors, we insert a random vector. That way we build an array [vocabulary_length x vector_dimension] and with the help of nn.Embedding() class from PyTorch we have the first layer of our model. Then, we preprocess the data from the vaccine_validation_set with the same way, but we do not build any Embedding layer. We just convert the words to a sequence of numbers. We pad and truncate our sequences so that they have equal number of words in order to be inserted to the embedding layer. The maximum number of words chosen was 25. It was an average number of the length of the tweets’ sentences and after some experimentation it was inferred that it was a good approach. With larger number of words per sentence, we observed overfitting in the validation set.

*	Then, we define the neural network, the hyper parameters and we train the model for 10 - 15 epochs with a specific batch size. At the end of every epoch we calculate the loss using the CrossEntropyLossFunction() and we evaluate our model providing the data of the validation_data_set, so that it can make predictions for the given data; which class every instance should have. As optimizer we use Adam, as it was suggested, and we provide to it learning rate = 0.0003 and weight_deacay = 0.0001 just to boost it a bit and it seems to work quite good with all the experiments. 

*	Finally, we plot the Learning Curves with the train_data_set and the validation_data_set with respect to every epoch to see how well the validation set has learned. We plot also the Roc curve to see the dependency between the true positive ratio and the false positive ratio. Moreover, we calculate the metrics of our model at each case (precision, recall, f1score, accuracy), so that we can evaluate how good the classifier is. At some cases, precision, recall and f1score are calculated separately for each class in order to evaluate how well or badly the model predicts a specific class.

*	We have experimented with several variants of each of the above ways and some of the results are presented below.

### Best model
After some experimentation, which is shown afterwards, we concluded that our best model has the following characteristics, using CrossEntropyLoss as Loss Function and Adam optimizer(learning_rate = 0.003, weight_decay = 1e-4):

*	Number of hidden layers = 64
*	Batch size = 32
*	Number of stacked layers = 2
*	Cell Type = GRU
*	Dropout probability = 0.2
*	Gradient clipping (max_norm = 1.0)
*	Learning Rate = 0.0004

It achieves f1score = 61% and accuracy = 69%, while the rest of the scores are shown below, as well as the learning curve and the ROC Curve. We can see that our model learns quite well at the beginning but it starts to overfit after the 10th epoch. Results at different executions may differ because of the dropout_probability, which alters the result each time. From the ROC Curve we observe that the most accurate predictions are made for class 0, which also has the most instances, then for class 2 and the least correct are for class 1, which has much less instances than the other two classes. The respective graphs are found under the *BEST MODELS -> RUN THE LSTM MODEL* header in the *Recurrent Neural Network* notebook.

### Experiments 
**(Except for the 1st one, all the other points apply to both LSTM and GRU RNNs. Note: All RNNs are bidirectional.)**

*	LSTM or GRU:

    It is hard to provide an accurate answer on this question, because both LSTM and GRU RNNs had similar performance when applying the same hyper parameters to both of them. We could say that when using LSTM, the model overfits quicker, and when using GRU, we achieve better learning Curve. The precision, recall, f1scores are similar, but they are slightly higher when using GRU implementation. Moreover, LSTM takes longer to train because it has cell states instead of only hidden states and the computations are more.

*	Number of Epochs & Parameters for Adam optimizer:

    We used maximum 10-15 epochs to train and evaluate each of our experiments, because we observed that the model no matter what alterations we applied was overfitting after the 8th – 9th epoch. Moreover, training and evaluation would take too long with a larger number of epochs and the final results would not be that different (or better) than the ones with less epochs.  After some experimentation, we concluded that the values for the Adam’s parameters should be: learning_rate =0.003 and weight_deacy = 1e-4, in order to apply some regularization and smoothing when computing the gradients.

*	Truncating and Padding:

    We pad the sentences of the tweet data in order to feed them in the RNNs. After some experimentation, we observed -given the corresponding data – that we should pad the sentences at maximum 20-25 words, because otherwise there are sentences with lots of zeros (they may have 5 or 6 words and if we padded the sentences at 40 words for example, this results to more computations for no reason and worse results – the model overfits very quickly). 
    
    When we pad the sentences at 20 words, with 128 hidden layers, 4 stacked layers, dropout_probability = 0.2 and batch_size = 64, we achieve with LSTM f1score = 49% and accuracy = 58% and the model begins to overfit after the 10th epoch, and with GRU we achieve f1score = 45% and accuracy = 55% and the model begins to overfit after the 7th epoch. We can see from the ROC curves also, how bad the performance of the models is.
  

*	Number of Hidden Layers:

    Here, because we have relatively small sentences on average and not a very large set of data (neither for training data, nor for validation data) a very big and complex structure of a RNN, produces worse results. So, we reduce the number of hidden layers at 80 and we set dropout_probability = 0, and we pad the sentences at maximum 15 words, leaving everything else as above. We observe the following:

    * **LSTM:** f1score = 51%, accuracy = 60%, an increase by 2% but the model still overfits, here after the 7th epoch. This is obvious from the learning curve. The ROC curve is obviously better than the previous ones.
             
    * **GRU:** f1score = 52%, accuracy = 61%, the increase is large; 6-7% but the model still overfits, here after the 7th epoch. This is obvious from the learning curve. The ROC curve is obviously better than the previous ones.

         
*	Number of Stacked Layers:
    
    Now, we reduce the number of stacked layers at 2 and we reduce the number of hidden layers a bit more at 48.

    * **LSTM:** There is a little improvement, f1score = 51%, accuracy = 61%, but this time precision is also higher. Now, it is 58%, whereas previously was 54%. This can be seen at the ROC Curve, which is enhanced. The lines for the 3 classes tend to move away from the center of the diagram.

    * **GRU:** f1score = 49% (lower than previously) and accuracy = 62%. It is obvious that the best predictions are made for class 2.

    The models now are not overfitting the validation set until the 10th epoch, as we can see from the learning curves.
       


*	Batch Size

    We reduce the batch size from 64 to 32 to feed into the RNNs smaller sequence of data to achieve better results. What we observe is:
    * **LSTM:** The total f1-score goes at 52% and the accuracy at 61%, which is good, however the model starts overfitting after the 7th epoch. 
    * **GRU:** The total f1-score goes at 53% and the accuracy at 62%, which is good, however the model starts overfitting after the 7th epoch.
    
    Precision and recall values are better for the GRU model example.


*	With Or Without Dropout Probability & Gradient Clipping:

    We will make an attempt to reduce the overfitting and maybe increase our scores by applying dropout probability after each training step and clipping the gradients during backpropagation so that they never exceed some threshold in order to make our model generalizing more easily. Both are regularization techniques and we expect for our model to have a better performance than without them. We use them to mitigate the exploding gradients problem. We set dropout_probability = 0.2 and max_norm = 2.0. The observations were the following:

    * **LSTM:** f1score = 53% & accuracy = 61%
    * **GRU:** f1score = 51% and accuracy = 61%

    Precision and recall has similar values for both RNNs (~52%-53%). There was not much improvement than previously, but the results are not worse, so we can infer that the techniques work equally well.

### CONCLUSION

Comparing the best version of the LSTM and GRU models, we have to admit that they perform equally well more or less. It depends both on the data sets and on the choice of the technic and the selection of the hyper parameters. LSTM implementation is definitely slower but in our case it turned out to be slightly better. It depends on the dataset and our needs. The implementation of the simple Neural Network with TF-Idf Vectorizer for this specific problem and dataset seemed to be the best choice. It was the fastest and it provided the best results. It gave high F1 scores and more correct predictions generally. The performance of the Softmax Regression model was a bit worse but it was still a good choice. The RNN model with GRU was faster than with LSTM.

## BERT-base model by [Huggingface](https://huggingface.co/bert-base-uncased)

BERT is a language representation model and stands for Bidirectional Encoder Representations from Transformers. BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. The pre-trained BERT model is used for a wide range of tasks, such as question answering.

### WORKFLOW OF THE PROJECT

*	The first step is to load our data into our project. So, the two given data sets are loaded; the vaccine_train_set, which will be used for the training of the neural network and the vaccine_validation_set, which will be used for evaluating the model. 

*	The second step is to preprocess these data, before they are inserted into the neural network. We clean the data by removing the “@example” annotations, the “” links and every sort of tokens, such as #, (, ), -, _, :, ;, etc. We also remove stop_words, because they may not be useful for our neural network since they are likely to appear many times in the data.

*	The next step is to create a dataset in order for our data more to be more organized and feasible to be split into batches. All the tokenization needed is applied inside the dataset and we use the BertTokenizerFast to tokenize the sentences. We follow the same process for both sets (vaccine_training_set and vaccine_validation_set). We pad and truncate our sequences so that they have equal number of words in order to be inserted to the embedding layers of BERT. The maximum number of words chosen was 25. It was an average number of the length of the tweets’ sentences and after some experimentation it was inferred that it was a good approach. With larger number of words per sentence, we observed overfitting in the validation set. 

*	Then, we define the BERT model (for more details see the jupyter notebook), the hyper parameters and we train the model for 4-5 epochs with a specific batch size. At the end of every epoch we calculate the loss using the CrossEntropyLossFunction() and we evaluate our model providing the data of the validation_data_set, so that it can make predictions for the given data; which class every instance should have. As optimizer we use AdamW, and we provide to it learning rate = 0.00005 and weight_deacay = 0.0001 just to boost it a bit and it seems to enhance the performance of all the experiments. 

*	Finally, we plot the Learning Curves with the train_data_set and the validation_data_set with respect to every epoch to see how well the validation set has learned. We plot also the Roc curve to see the dependency between the true positive ratio and the false positive ratio. Moreover, we calculate the metrics of our model at each case (precision, recall, f1score, accuracy), so that we can evaluate how good the classifier is. At some cases, precision, recall and f1score are calculated separately for each class in order to evaluate how well or badly the model predicts a specific class.

*	We have experimented with several variants of each of the above ways and some of the results are presented below.

### Best model

After some experimentation, which is shown afterwards, we concluded that our best model has the following characteristics, using CrossEntropyLoss as Loss Function and AdamW optimizer(learning_rate = 0.003, weight_decay = 1e-4):

*	Batch size = 32
*	Dropout probability = 0.2
*	Learning Rate = 0.00005
*	After the BertModel.from_pretrained(‘bert-base-uncased’) there is a sequence of additional layers to enhance the classification process. So, we have a linear layer, then a ReLU activation function, then a dropout layer with probability 0.2 (as mentioned above), then another linear layer and finally a Softmax Function in order to get the probabilities for each class and then take the class with the highest probability. 
*	Max Tokens = 25
*	Time for training/ evaluation per epoch = 3 minutes

It achieves f1score = 48% and accuracy = 67%, while the rest of the scores are shown below, as well as the learning curve and the ROC Curve. We can see that our model does not really learn well during these 5 epochs and there is an overfitting from the beginning. Results at different executions may differ because of the dropout_probability, which alters the result each time. From the ROC Curve we observe that the most accurate predictions are made for class 0, which also has the most instances, then for class 2 and finally for class 1, the model cannot predict any instances, which may be owed to the fact that this class has much less instances than the other two classes. The respective graphs are found in the *BERT_Model* notebook under the *EXECUTION -> FINAL MODEL* header:


### Experiments 

*	Number of Epochs & Parameters for AdamW optimizer:
    
    We used maximum 3-5 epochs to train and evaluate each of our experiments, because we observed that the model no matter what alterations we applied was overfitting after the 4th – 5th epoch. Moreover, training and evaluation would take too long with a larger number of epochs and the final results would not be that different (or better) than the ones with less epochs.  After some experimentation, we concluded that the values for the AdamW’s parameters should be: learning_rate = 0.00005 and weight_deacy = 1e-4, in order to apply some regularization and smoothing when computing the gradients.

*	Learning rate = 0.00001, batch_size=16,  time = 6min/epoch, max_tokens = 50:
    
    We set the number of maximum tokens at each sequence = 50 and small batch size and we observe that our model achieves an accuracy of 64% and the f1score is 45%. The model cannot predict the class 1, maybe because it has much less instances (294) than the other two classes (1060, 918). From the learning curve we can definitely see an overfit in the validation set and from the ROC curve we understand the model predicts more easily instances of class 0.

*	Learning rate = 0.00001, batch_size=16,  time = 4.5 min/epoch, max_tokens = 35:
    
    We decrease the number of the tokens a bit and the only thing which seems to change is the total time needed for training and evaluation. Here we need 4.5 min/epoch whereas we needed 6 min/ epoch above. This is owed to the smaller number of tokens.

*	Learning rate = 0.00001, batch_size=16,  time = 4 min/epoch, max_tokens = 25:
    
    We observe similar things as in the previous two experiments. Only training and evaluation time per epoch decreases. Our model achieves an accuracy of 62% and the f1score is 44%. The model cannot still predict the class 1. From the learning curve we can definitely see an overfit in the validation set and from the ROC curve is worse than the ones previously.

*	Learning rate = 0.0001, batch_size=32,  time = 6 min/epoch, max_tokens = 50:
    
    We increased the learning rate to 1e-4 and the batch size to 32 and we kept the maximum number of tokens per sequence at 50. We needed 6 min/epoch for training and evaluation. We can see from the graphs that the learning curve is not smooth at all, but the overfitting seems to be smaller. The model achieves an accuracy of 64% and the f1score is 46%. The metrics increased a bit. The inferences from the ROC curve remain the same as previously.


*	Learning rate = 0.00005, batch_size=32,  time = 3.5 min/epoch, max_tokens = 35:

    We decreased the learning rate to 5e-5 and we kept the batch size to 32 and we reduced the maximum number of tokens per sequence at 35. We needed 3.5 min/epoch for training and evaluation. We can see from the graphs that in the learning curve there is definitely an overfitting at the validation set. The model achieves an accuracy of 63% and the f1score is 45%. The values of the metrics remain at similar levels. The inferences from the ROC curve remain the same as previously. The most correct predictions are for class 0, then for class 2 and finally for class 1 the model cannot predict any correct instances.


## Comparison of the models using the 4 techniques

After experimenting with trying to implement a sentiment classifier using these 4 different techniques, we have made some inferences.

Models with Softmax Regression and Feed Forward Neural Network turned out to perform better than the models implemented with Recurrent Neural Networks and those trained with the pre-trained BertModel. This could be owed to the fact that these models use CountVectorizer and TF-IDFVectorizer respectively in order to build their vocabulary and they do not use pre-trained embeddings as the latter one. The former two models seem to be equally good and their performances are very similar to each other. The model using the Feed Forward neural network seems to achieve 1% - 2% higher scores at all metrics but this is a minor difference. The f1score of the model using the Feed Forward neural network is 66% and the accuracy is 72%. The f1score of the model using softmax regression is 65% and the accuracy is 70%. The accuracy for the RNN model with GRU cells is 69%; lower than the accuracy of the other two models and the total f1 score is 61%. The accuracy of the model trained with BERT is 67% and the f1score is 48%; a lot lower than the other 3 models.

We could say that only the precision of class 1 is a lot better at the model using the neural network; it is 61%, whereas at the model using softmax regression is 55% and at the GRU model is 38%. So, the model with neural network classifies more instances of class 1 as class 1 than the model with softmax regression and the GRU model. The GRU model cannot identify very well instances of class 1. The model using BERT cannot identify any instances of class 1. For class 0, the performance of the 3 models is almost the same. All the metrics (precision, recall, f1_score) have values ~77%. For class 2, the best performance is that of the model with Neural Network but with minor difference from the other 2. Precision and recall generally for the 3 classes are higher for the model with Neural Network, then follows the model with Softmax Regression and the lowest are for the RNN – GRU model.

We needed 100 epochs to achieve the above results for the models with Softmax Regression and Feed Forward Neural Network, and only 10 epochs for the RNN model with GRU, because after the 10th the model overfits. For the model with BERT we used only 5 epochs to train and evaluate the model because it takes too much time otherwise. Maybe, more data would have made the learning process better for the GRU model. So, we should recognize the power of the RNN model as well as of BERT model, since they need only 10% and 5% of the epochs respectively to achieve a slightly worse performance than the other two models. The metrics were ~7% - 8% lower than the metrics of the other two models.

## BERT for SQuAD 2.0
We developed a model for question answering on
the dataset SQuAD 2.0. In the [SQuAD explorer](https://rajpurkar.github.io/SQuAD-explorer/) Web site there is the dataset and relevant information in order to fine-tune BERT-base for this task.

### WORKFLOW OF THE PROJECT

* The first step is to load our data into our project. So, the two data sets from “https://rajpurkar.github.io/SQuAD-explorer/” are loaded; the “train-v2.0.json”, which will be used for fine-tuning the bert_model for question answering and the "dev-v2.0.json", which will be used for evaluating the model. 

* We create a dataframe with columns = ['context', 'question', 'answer', 'start_index', 'end_index'] and we remove any duplicate rows. We add the end_index accordingly, as described in the notebook, when an answer is available. When a question does not have an answer we apply the same value for both the start_index and end_index. 

*	The next step is to convert the words of the sentences into tokens. For the tokenization we use the BertTokenizerFast to tokenize the sentences. We follow the same process for both sets (training_set and test_set). We pad and truncate our sequences so that they have equal number of words in order to be inserted to the embedding layers of BERT. The maximum number of words chosen was 350. When we used more tokens, we were left out of RAM in colab.

*	Then, we define the BERT model for question answering (for more details see the jupyter notebook), the hyper parameters and we train the model for 3 epochs with a specific batch size. At the end of every epoch we calculate the loss and we evaluate our model providing the data of the data_set, so that it can make predictions for the given data; which is the answer, which answers a specific question. As optimizer we use AdamW, and we provide to it learning rate = 0.00005 and weight_deacay = 0.0001 just to boost it a bit and it seems to enhance the performance of all the experiments.  The whole process took about 4 hours. The difficulties and struggles encountered until we reach this solution are described in the notebook.

*	Finally, we plot the Learning Curves with the train_data_set and the validation_data_set with respect to every epoch to see how well the validation set has learned. 

*	We have experimented with several variants and details about these you can find in the notebook.

*	We save the model so that we can use it any time we want.

*	Finally, we use the data from “dev-v2.0.json” file in order to calculate the Exact Matches (EM) scores and F1 scores. We have an exact match when the answer provided by the model is exactly the same with the answer provided by the dataset. We used the Evaluation Script (some of the functions which are in this file) provided by Stanford in order to apply some normalization for the answers (make letters small, remove tokens etc).

Our model got correct 5668 / 16315 sentences. The f1 score for the whole dataset is: 0.4652