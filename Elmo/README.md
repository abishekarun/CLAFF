## Deep Contextualized Word vectors

This directory contains all the models ran with elmo embeddings for both the subtasks. The models were all LSTM with attention and we did semi supervised learning wherein we added most confident predictions of our model to our dataset and trained again. We also present visualization results of our attention to understand our models better. 

The model results for social can be found [here](https://github.com/abishekarun/CLAFF/blob/master/Elmo/Social/social_results.txt) and agency [here](https://github.com/abishekarun/CLAFF/blob/master/Elmo/Agency/agency_results.txt). The visualization results can be found in this [directory](https://github.com/abishekarun/CLAFF/blob/master/Elmo/Visualization/).

The resources that helped me are:

+ [ELMo](https://allennlp.org/elmo)
+ [Contextual Language Embedding](https://towardsdatascience.com/elmo-contextual-language-embedding-335de2268604)
+ [ELMo Illustration](http://jalammar.github.io/illustrated-bert/)
+ [ELMo for extracting features](https://www.analyticsvidhya.com/blog/2019/03/learn-to-use-elmo-to-extract-features-from-text/)
