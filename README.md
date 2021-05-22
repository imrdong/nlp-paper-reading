# NLP-Paper-Reading

![Language Markdown](https://img.shields.io/badge/Language-Markdown-red)
[![License CC0-1.0](https://img.shields.io/badge/License-CC0--1.0-blue.svg)](https://github.com/imrdong/nlp-paper-reading/blob/master/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/imrdong/nlp-paper-reading.svg?style=social&label=Star&maxAge=10)](https://github.com/imrdong/nlp-paper-reading/stargazers/)
[![GitHub forks](https://img.shields.io/github/forks/imrdong/nlp-paper-reading?style=social&label=Fork&maxAge=10)](https://github.com/imrdong/nlp-paper-reading/network/members/)

## Paper Reading List in Natural Language Processing  

## 🧐 Table of Contents

- [Dialogue System](#dialogue-system)
    - [Background Based Conversation](#background-based-conversation)
    - [Dataset for Dialogue](#dataset-for-dialogue)
    - [Emotional Dialogue](#emotional-dialogue)
    - [Evaluation of Dialogue](#evaluation-of-dialogue) 
    - [Open-domain Dialogue](#open-domain-dialogue)
    - [Personalized Dialogue](#personalized-dialogue)  
    - [Survey on Dialogue](#survey-on-dialogue)
    - [Task-oriented Dialogue](#task-oriented-dialogue)
- [Pre-trained Language Model](#pre-trained-language-model)  
- [Text Summarization](#text-summarization)
    - [Abstractive](#abstractive)
- [Transformer Series](#transformer-series)
- [Word Embedding](#word-embedding)

## Dialogue System

### Background Based Conversation

* **GLKS**: Thinking Globally, Acting Locally: Distantly Supervised Global-to-Local Knowledge Selection for Background Based Conversation. AAAI-2020 [[pdf]](https://ojs.aaai.org//index.php/AAAI/article/view/6395) [[code]](https://github.com/PengjieRen/GLKS)
* **RefNet**: RefNet: A Reference-Aware Network for Background Based Conversation. AAAI-2020 [[pdf]](https://ojs.aaai.org//index.php/AAAI/article/view/6370) [[code]](https://github.com/ChuanMeng/RefNet)
* **CaKe**: Improving Background Based Conversation with Context-aware Knowledge Pre-selection. arXiv-2019 [[pdf]](https://arxiv.org/abs/1906.06685) [[code]](https://github.com/repozhang/bbc-pre-selection)

### Dataset for Dialogue

* **MuTual**: A Dataset for Multi-Turn Dialogue Reasoning. ACL-2020 [[pdf]](https://www.aclweb.org/anthology/2020.acl-main.130.pdf) [[data]](https://github.com/Nealcly/MuTual)
* **DailyDialog**: DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset. IJCNLP-2017 [[pdf]](https://www.aclweb.org/anthology/I17-1099.pdf) [[data]](http://yanran.li/dailydialog)  
* **Douban Conversation Corpus**: Sequential Matching Network: A New Architecture for Multi-turn Response Selection in Retrieval-Based Chatbots. ACL-2017 [[pdf]](https://www.aclweb.org/anthology/P17-1046.pdf) [[data]](https://github.com/MarkWuNLP/MultiTurnResponseSelection)  
* **Ubuntu Dialogue Corpus**: The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems. SIGDIAL-2015 [[pdf]](https://www.aclweb.org/anthology/W15-4640.pdf) [[data]](http://www.cs.mcgill.ca/~jpineau/datasets/ubuntu-corpus-1.0)  

### Emotional Dialogue

* **MOJITALK**: MOJITALK: Generating Emotional Responses at Scale. ACL-2018 [[pdf]](https://www.aclweb.org/anthology/P18-1104.pdf)

### Evaluation of Dialogue

* **RS**: Evaluating Dialogue Generation Systems via Response Selection. ACL-2020 [[pdf]](https://www.aclweb.org/anthology/2020.acl-main.55.pdf) [[test set]](https://github.com/cl-tohoku/eval-via-selection)
* **BLEU**: BLEU: a Method for Automatic Evaluation of Machine Translation. ACL-2002 [[pdf]](https://www.aclweb.org/anthology/P02-1040.pdf)

### Open-domain Dialogue

* **Adp Multi-CL**: Learning from Easy to Complex: Adaptive Multi-Curricula Learning for Neural Dialogue Generation. AAAI-2020 [[pdf]](https://aaai.org/ojs/index.php/AAAI/article/view/6244)
* **DialogRPT**: Dialogue Response Ranking Training with Large-Scale Human Feedback Data. EMNLP-2020 [[pdf]](https://www.aclweb.org/anthology/2020.emnlp-main.28.pdf) [[code]](https://github.com/golsun/DialogRPT)
* **Low-Resource KG**: Low-Resource Knowledge-Grounded Dialogue Generation. ICLR-2020 [[pdf]](https://openreview.net/pdf?id=rJeIcTNtvS)
* **PS**: Prototype-to-Style: Dialogue Generation with Style-Aware Editing on Retrieval Memory. arXiv-2020 [[pdf]](https://arxiv.org/pdf/2004.02214.pdf)
* **Edit-N-Rerank**: Response Generation by Context-aware Prototype Editing. AAAI-2019 [[pdf]](https://ojs.aaai.org//index.php/AAAI/article/view/4714) [[code]](https://github.com/MarkWuNLP/ResponseEdit)
* **PostKS**: Learning to Select Knowledge for Response Generation in Dialog Systems. IJCAI-2019 [[pdf]](https://www.ijcai.org/proceedings/2019/0706.pdf) [[code]](https://github.com/bzantium/Posterior-Knowledge-Selection)
* **SeqGen**: Retrieval-guided Dialogue Response Generation via a Matching-to-Generation Framework. EMNLP-2019 [[pdf]](https://www.aclweb.org/anthology/D19-1195.pdf) [[code]](https://github.com/jcyk/seqgen)
* **SR**: Skeleton-to-Response: Dialogue Generation Guided by Retrieval Memory. NAACL-2019 [[pdf]](https://www.aclweb.org/anthology/N19-1124.pdf)
* **NKD**: Knowledge Diffusion for Neural Dialogue Generation. ACL-2018 [[pdf]](https://www.aclweb.org/anthology/P18-1138.pdf) [[code]](https://github.com/liushuman/neural-knowledge-diffusion)
* **CVAE**: Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders. ACL-2017 [[pdf]](https://www.aclweb.org/anthology/P17-1061.pdf)
* **DRL**: Deep Reinforcement Learning for Dialogue Generation. EMNLP-2016 [[pdf]](https://www.aclweb.org/anthology/D16-1127.pdf)
* **MMI**: A Diversity-Promoting Objective Function for Neural Conversation Models. NAACL-2016 [[pdf]](https://www.aclweb.org/anthology/N16-1014.pdf) 
* **NCM**: A Neural Conversational Model. arXiv-2015 [[pdf]](https://arxiv.org/pdf/1506.05869.pdf)

### Personalized Dialogue

* **GCC**: Large Scale Multi-Actor Generative Dialog Modeling. ACL-2020 [[pdf]](https://www.aclweb.org/anthology/2020.acl-main.8.pdf)
* **GDR**: Generate, Delete and Rewrite: A Three-Stage Framework for Improving Persona Consistency of Dialogue Generation. ACL-2020 [[pdf]](https://www.aclweb.org/anthology/2020.acl-main.516.pdf)
* **KvPI**: Proﬁle Consistency Identiﬁcation for Open-domain Dialogue Agents. EMNLP-2020 [[pdf]](https://www.aclweb.org/anthology/2020.emnlp-main.539.pdf) [[code]](https://github.com/songhaoyu/KvPI)
* **P<sup>2</sup> Bot**: You Impress Me: Dialogue Generation via Mutual Persona Perception. ACL-2020 [[pdf]](https://www.aclweb.org/anthology/2020.acl-main.131.pdf) [[code]](https://github.com/SivilTaram/Persona-Dialogue-Generation)
* **PAGenerator**: Guiding Variational Response Generator to Exploit Persona. ACL-2020 [[pdf]](https://www.aclweb.org/anthology/2020.acl-main.7.pdf)
* **Persona-sparse**: A Pre-training Based Personalized Dialogue Generation Model with Persona-sparse Data. AAAI-2020 [[pdf]](https://ojs.aaai.org//index.php/AAAI/article/view/6518)
* **Self-Consciousness**: Will I Sound Like Me? Improving Persona Consistency in Dialogues through Pragmatic Self-Consciousness. EMNLP-2020 [[pdf]](https://www.aclweb.org/anthology/2020.emnlp-main.65.pdf) [[code]](https://github.com/skywalker023/pragmatic-consistency)
* **PAML**: Personalizing Dialogue Agents via Meta-Learning. ACL-2019 [[pdf]](https://www.aclweb.org/anthology/P19-1542.pdf) [[code]](https://github.com/HLTCHKUST/PAML)
* **PerCVAE**: Exploiting Persona Information for Diverse Generation of Conversational Responses. IJCAI-2019 [[pdf]](https://www.ijcai.org/Proceedings/2019/0721.pdf) [[code]](https://github.com/vsharecodes/percvae)
* **PerDG**: Personalized Dialogue Generation with Diversified Traits. arXiv-2019 [[pdf]](https://arxiv.org/pdf/1901.09672.pdf)
* **NPRG-DM**: Neural personalized response generation as domain adaptation. WWW-2018 [[pdf]](https://link.springer.com/content/pdf/10.1007/s11280-018-0598-6.pdf)
* **PCCM**: Assigning Personality Profile to a Chatting Machine for Coherent Conversation Generation. IJCAI-2018 [[pdf]](https://www.ijcai.org/Proceedings/2018/0595.pdf)
* **PerDA**: Training Millions of Personalized Dialogue Agents. EMNLP-2018 [[pdf]](https://www.aclweb.org/anthology/D18-1298.pdf)
* **PERSONA-CHAT**: Personalizing Dialogue Agents: I have a dog, do you have pets too? ACL-2018 [[pdf]](https://www.aclweb.org/anthology/P18-1205.pdf)
* **CoPerHED**: Exploring Personalized Neural Conversational Models. IJCAI-2017 [[pdf]](https://www.ijcai.org/proceedings/2017/0521.pdf)
* **PRG-DM**: Personalized Response Generation via Domain adaptation. SIGIR-2017 [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3077136.3080706)
* **PerNCM**: A Persona-Based Neural Conversation Model. ACL-2016 [[pdf]](https://www.aclweb.org/anthology/P16-1094.pdf)
* **PERSONAGE**: PERSONAGE: Personality Generation for Dialogue. ACL-2007 [[pdf]](https://www.aclweb.org/anthology/P07-1063.pdf)

### Survey on Dialogue

* **Challenges**: Challenges in Building Intelligent Open-domain Dialog Systems. TOIS-2020 [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3383123)
* **Recipes**: Recipes for building an open-domain chatbot. arXiv-2020 [[pdf]](https://arxiv.org/pdf/2004.13637.pdf)
* **Recent Advances and New Frontiers**: A Survey on Dialogue Systems: Recent Advances and New Frontiers. SIGKDD-2017 [[pdf]](https://www.kdd.org/exploration_files/19-2-Article3.pdf)

### Task-oriented Dialogue

* **DF-Net**: Dynamic Fusion Network for Multi-Domain End-to-end Task-Oriented Dialog. ACL-2020 [[pdf]](https://www.aclweb.org/anthology/2020.acl-main.565.pdf) [[code]](https://github.com/LooperXX/DF-Net)
* **MALA**: MALA: Cross-Domain Dialogue Generation with Action Learning. AAAI-2020 [[pdf]](https://aaai.org/ojs/index.php/AAAI/article/view/6306)
* **BossNet**: Disentangling Language and Knowledge in Task-Oriented Dialogs. NAACL-2019 [[pdf]](https://www.aclweb.org/anthology/N19-1126.pdf) [[code]](https://github.com/dair-iitd/BossNet)
* **GLMP**: GLOBAL-TO-LOCAL MEMORY POINTER NETWORKS FOR TASK-ORIENTED DIALOGUE. ICLR-2019 [[pdf]](https://openreview.net/pdf?id=ryxnHhRqFm) [[code]](https://github.com/jasonwu0731/GLMP)
* **KB Retriever**: Entity-Consistent End-to-end Task-Oriented Dialogue System with KB Retriever. EMNLP-2019 [[pdf]](https://www.aclweb.org/anthology/D19-1013.pdf) [[data]](https://github.com/yizhen20133868/Retriever-Dialogue)
* **TRADE**: Transferable Multi-Domain State Generator for Task-Oriented Dialogue Systems. ACL-2019 [[pdf]](https://www.aclweb.org/anthology/P19-1078.pdf) [[code]](https://github.com/jasonwu0731/trade-dst)
* **DSR**: Sequence-to-Sequence Learning for Task-oriented Dialogue with Dialogue State Representation. COLING-2018 [[pdf]](https://www.aclweb.org/anthology/C18-1320.pdf)
* **Mem2Seq**: Mem2Seq: Effectively Incorporating Knowledge Bases into End-to-End Task-Oriented Dialog Systems. ACL-2018 [[pdf]](https://www.aclweb.org/anthology/P18-1136.pdf) [[code]](https://github.com/HLTCHKUST/Mem2Seq)
* **StateNet**: Towards Universal Dialogue State Tracking. EMNLP-2018 [[pdf]](https://www.aclweb.org/anthology/D18-1299.pdf)
* **TSCP**: Sequicity: Simplifying Task-oriented Dialogue Systems with Single Sequence-to-Sequence Architectures. ACL-2018 [[pdf]](https://www.aclweb.org/anthology/P18-1133.pdf) [[code]](https://github.com/WING-NUS/sequicity)

## Pre-trained Language Model

* **ALBERT**: ALBERT: A Lite BERT for Self-supervised Learning of Language Representations. ICLR-2020 [[pdf]](https://openreview.net/pdf?id=H1eA7AEtvS) [[code]](https://github.com/google-research/ALBERT)
* **BERT**: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-2019 [[pdf]](https://www.aclweb.org/anthology/N19-1423) [[code]](https://github.com/google-research/bert)
* **ERNIE**: ERNIE: Enhanced Language Representation with Informative Entities. ACL-2019 [[pdf]](https://www.aclweb.org/anthology/P19-1139.pdf) [[code]](https://github.com/thunlp/ERNIE)
* **Interpret_BERT**: What does BERT learn about the structure of language? ACL-2019 [[pdf]](https://www.aclweb.org/anthology/P19-1356.pdf) [[code]](https://github.com/ganeshjawahar/interpret_bert)
* **UniLM**: Unified Language Model Pre-training for Natural Language Understanding and Generation. NeurIPS-2019 [[pdf]](https://proceedings.neurips.cc/paper/2019/file/c20bb2d9a50d5ac1f713f8b34d9aac5a-Paper.pdf) [[code]](https://github.com/microsoft/unilm)
* **XLNet**: XLNet: Generalized Autoregressive Pretraining for Language Understanding. NeurIPS-2019 [[pdf]](http://papers.nips.cc/paper/8812-xlnet-generalized-autoregressive-pretraining-for-language-understanding.pdf) [[code]](https://github.com/zihangdai/xlnet)

## Text Summarization

### Abstractive

* **GTTP**: Get To The Point: Summarization with Pointer-Generator Networks. ACL-2017 [[pdf]](https://www.aclweb.org/anthology/P17-1099.pdf) [[code]](https://github.com/abisee/pointer-generator)

## Transformer Series

* **GRET**: GRET: Global Representation Enhanced Transformer. AAAI-2020 [[pdf]](https://aaai.org/ojs/index.php/AAAI/article/view/6464)
* **Reformer**: REFORMER: THE EFFICIENT TRANSFORMER. ICLR-2020 [[pdf]](https://openreview.net/pdf?id=rkgNKkHtvB) [[code]](https://github.com/google/trax/tree/master/trax/models/reformer)
* **Transformer-XL**: Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context. ACL-2019 [[pdf]](https://www.aclweb.org/anthology/P19-1285.pdf) [[code]](https://github.com/kimiyoung/transformer-xl)
* **Universal Transformers**: UNIVERSAL TRANSFORMERS. ICLR-2019 [[pdf]](https://openreview.net/pdf?id=HyzdRiR9Y7) [[code]](https://github.com/tensorflow/tensor2tensor)
* **Transformer**: Attention Is All You Need. NeurIPS-2017 [[pdf]](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) [[code-official]](https://github.com/tensorflow/tensor2tensor) [[code-tf]](https://github.com/Kyubyong/transformer) [[code-py]](https://github.com/jadore801120/attention-is-all-you-need-pytorch)

## Word Embedding

* **Doc2Vec**: Distributed Representations of Sentences and Documents. ICML-2014 [[pdf]](http://proceedings.mlr.press/v32/le14.pdf)
* **Glove**: GloVe: Global Vectors for Word Representation. EMNLP-2014 [[pdf]](https://www.aclweb.org/anthology/D14-1162.pdf) [[code]](https://github.com/stanfordnlp/GloVe)
* **Word2Vec-Extension**: Distributed Representations of Words and Phrases and their Compositionality. NIPS-2013 [[pdf]](https://proceedings.neurips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf)
* **Word2Vec-Oringinal**: Efficient Estimation of Word Representations in Vector Space. ICLR Workshop Poster-2013 [[pdf]](https://arxiv.org/pdf/1301.3781.pdf) [[code]](https://code.google.com/p/word2vec/)