## Contextualized Word Vectors

This directory contains all the models ran with CoVe embeddings for both the subtasks. We used the official source code that was released with the paper. It can be found [here](https://github.com/salesforce/cove).
The models were all LSTM with attention with 1 hidden layer and 1024 hidden units.

The model results for social can be found [here](https://github.com/abishekarun/CLAFF/blob/master/Cove/test/social/social_results.txt) and agency [here](https://github.com/abishekarun/CLAFF/blob/master/Cove/test/agency/agency_results.txt). 

To run the models, download the glove embeddings text file and mention the directory in the scripts when loading glove vectors. 

The resources that helped me are:

+ [Contextualized word vectors](https://towardsdatascience.com/replacing-your-word-embeddings-by-contextualized-word-vectors-9508877ad65d)
+ [Understanding word vectors](https://medium.com/explorations-in-language-and-learning/understanding-word-vectors-f5f9e9fdef98)
+ [Generalized language models](https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html)
+ [Contextualized word representations](https://medium.com/@ayush2503/contextualized-word-representations-5df54663323f)
