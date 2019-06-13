## [CL-AFF SHARED TASK](https://sites.google.com/view/affcon2019/cl-aff-shared-task)

This work was done for the CLAFF shared task held as a part of the Affective Content Analysis workshop @ AAAI 2019.It comprises of two sub-tasks(binary classification problems of 'agency' and 'sociality' attributes) for analyzing happiness and wellbeing in written language, on a corpus of 100,000 descriptions of happy moments(HappyDB corpus). 

The dataset can be found [here](https://github.com/abishekarun/CLAFF/blob/master/data/) and our paper can be found [here](https://arxiv.org/abs/1906.03677). Our Elmo model showed the best performance on both subtasks and won the first position.

|    **Model Directory**  | **READme file** |
|--------------------|------------|
| [Glove](https://github.com/abishekarun/CLAFF/blob/master/Glove/) | [1](https://github.com/abishekarun/CLAFF/blob/master/Glove/README.md)|
| [CoVe](https://github.com/abishekarun/CLAFF/blob/master/Cove/) | [2](https://github.com/abishekarun/CLAFF/blob/master/Cove/README.md) |
| [Elmo](https://github.com/abishekarun/CLAFF/blob/master/Elmo/) | [3](https://github.com/abishekarun/CLAFF/blob/master/Elmo/README.md) |
| [Fastai](https://github.com/abishekarun/CLAFF/blob/master/Fastai/) | [4](https://github.com/abishekarun/CLAFF/blob/master/Fastai/README.md) |

The resources that helped me are:

+ [RNN effectiveness](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
+ [Contextual Language Embedding](https://towardsdatascience.com/elmo-contextual-language-embedding-335de2268604)
+ [ELMo Illustration](http://jalammar.github.io/illustrated-bert/)
+ [Universal language models](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html)
+ [ULMFiT Tutorial](https://www.analyticsvidhya.com/blog/2018/11/tutorial-text-classification-ulmfit-fastai-library/)
+ [Fastai in production](https://hackernoon.com/fast-ai-in-production-real-word-text-classification-with-ulmfit-199769be2a6)