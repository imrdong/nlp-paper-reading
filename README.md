# NLP-Paper-Reading

![Language Markdown](https://img.shields.io/badge/Language-Markdown-red)
[![License CC0-1.0](https://img.shields.io/badge/License-CC0--1.0-blue.svg)](https://github.com/imrdong/nlp-paper-reading/blob/master/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/imrdong/nlp-paper-reading.svg?style=social&label=Star&maxAge=10)](https://github.com/imrdong/nlp-paper-reading/stargazers/)
[![GitHub forks](https://img.shields.io/github/forks/imrdong/nlp-paper-reading?style=social&label=Fork&maxAge=10)](https://github.com/imrdong/nlp-paper-reading/network/members/)

## Paper Reading List in Natural Language Processing  

## 🧐 Table of Contents

- [Bert Series](#bert-series)
- [Dialogue System](#dialogue-system)
    - [Dataset for Dialogue](#dataset-for-dialogue)
    - [Emotional Dialogue](#emotional-dialogue)
    - [Evaluation of Dialogue](#evaluation-of-dialogue) 
    - [Open-domain Dialogue](#open-domain-dialogue)
    - [Personalized Dialogue](#personalized-dialogue)  
    - [Survey on Dialogue](#survey-on-dialogue)
    - [Task-oriented Dialogue](#task-oriented-dialogue)  
- [Transformer Series](#transformer-series)

## Bert Series

* **ALBERT**: ALBERT: A Lite BERT for Self-supervised Learning of Language Representations. ICLR-2020 [[pdf]](https://openreview.net/pdf?id=H1eA7AEtvS) [[code]](https://github.com/google-research/ALBERT)
* **BERT**: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-2019 [[pdf]](https://www.aclweb.org/anthology/N19-1423) [[code]](https://github.com/google-research/bert)
* **ERNIE**: ERNIE: Enhanced Language Representation with Informative Entities. ACL-2019 [[pdf]](https://www.aclweb.org/anthology/P19-1139.pdf) [[code]](https://github.com/thunlp/ERNIE)
* **XLNet**: XLNet: Generalized Autoregressive Pretraining for Language Understanding. NeurIPS-2019 [[pdf]](http://papers.nips.cc/paper/8812-xlnet-generalized-autoregressive-pretraining-for-language-understanding.pdf) [[code]](https://github.com/zihangdai/xlnet)

## Dialogue System

### Dataset for Dialogue

* **MuTual**: A Dataset for Multi-Turn Dialogue Reasoning. ACL-2020 [[pdf]](https://www.aclweb.org/anthology/2020.acl-main.130.pdf) [[data]](https://github.com/Nealcly/MuTual)
* **DailyDialog**: DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset. IJCNLP-2017 [[pdf]](https://www.aclweb.org/anthology/I17-1099.pdf) [[data]](http://yanran.li/dailydialog)  

### Emotional Dialogue

* **MOJITALK**: MOJITALK: Generating Emotional Responses at Scale. ACL-2018 [[pdf]](https://www.aclweb.org/anthology/P18-1104.pdf)

### Evaluation of Dialogue

* **RS**: Evaluating Dialogue Generation Systems via Response Selection. ACL-2020 [[pdf]](https://www.aclweb.org/anthology/2020.acl-main.55.pdf) [[test set]](https://github.com/cl-tohoku/eval-via-selection)
* **BLEU**: BLEU: a Method for Automatic Evaluation of Machine Translation. ACL-2002 [[pdf]](https://www.aclweb.org/anthology/P02-1040.pdf)

### Open-domain Dialogue

* **Adp Multi-CL**: Learning from Easy to Complex: Adaptive Multi-Curricula Learning for Neural Dialogue Generation. AAAI-2020 [[pdf]](https://aaai.org/ojs/index.php/AAAI/article/view/6244)
* **Low-Resource KG**: Low-Resource Knowledge-Grounded Dialogue Generation. ICLR-2020 [[pdf]](https://openreview.net/pdf?id=rJeIcTNtvS)
* **PS**: Prototype-to-Style: Dialogue Generation
with Style-Aware Editing on Retrieval Memory. ACL-2020 [[pdf]](https://arxiv.org/pdf/2004.02214.pdf)
* **RefNet**: A Reference-aware Network for Background Based Conversation. AAAI-2020 [[pdf]](https://arxiv.org/pdf/1908.06449.pdf) [[code]](https://github.com/ChuanMeng/RefNet)
* **Edit-N-Rerank**: Response Generation by Context-aware Prototype Editing. AAAI-2019 [[pdf]](https://arxiv.org/pdf/1806.07042.pdf) [[code]](https://github.com/MarkWuNLP/ResponseEdit)
* **PostKS**: Learning to Select Knowledge for Response Generation in Dialog Systems. IJCAI-2019 [[pdf]](https://www.ijcai.org/proceedings/2019/0706.pdf) [[code]](https://github.com/bzantium/Posterior-Knowledge-Selection)
* **SeqGen**: Retrieval-guided Dialogue Response Generation via a Matching-to-Generation Framework. EMNLP-2019 [[pdf]](https://www.aclweb.org/anthology/D19-1195.pdf) [[code]](https://github.com/jcyk/seqgen)
* **SR**: Skeleton-to-Response: Dialogue Generation Guided by Retrieval Memory. NAACL-2019 [[pdf]](https://www.aclweb.org/anthology/N19-1124.pdf)
* **CVAE**: Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders. ACL-2017 [[pdf]](https://www.aclweb.org/anthology/P17-1061.pdf)
* **DRL**: Deep Reinforcement Learning for Dialogue Generation. EMNLP-2016 [[pdf]](https://www.aclweb.org/anthology/D16-1127.pdf)
* **MMI**: A Diversity-Promoting Objective Function for Neural Conversation Models. NAACL-2016 [[pdf]](https://www.aclweb.org/anthology/N16-1014.pdf) 
* **NCM**: A Neural Conversational Model. arXiv-2015 [[pdf]](https://arxiv.org/pdf/1506.05869.pdf)

### Personalized Dialogue

* **GDR**: Generate, Delete and Rewrite: A Three-Stage Framework for Improving Persona Consistency of Dialogue Generation. ACL-2020 [[pdf]](https://www.aclweb.org/anthology/2020.acl-main.516.pdf)
* **P<sup>2</sup> Bot**: You Impress Me: Dialogue Generation via Mutual Persona Perception. ACL-2020 [[pdf]](https://www.aclweb.org/anthology/2020.acl-main.131.pdf) [[code]](https://github.com/SivilTaram/Persona-Dialogue-Generation)
* **PAGenerator**: Guiding Variational Response Generator to Exploit Persona. ACL-2020 [[pdf]](https://www.aclweb.org/anthology/2020.acl-main.7.pdf)
* **Persona-sparse**: A Pre-training Based Personalized Dialogue Generation Model with Persona-sparse Data. AAAI-2020 [[pdf]](https://arxiv.org/pdf/1911.04700.pdf)
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
* **GLMP**: GLOBAL-TO-LOCAL MEMORY POINTER NETWORKS FOR TASK-ORIENTED DIALOGUE. ICLR-2019 [[pdf]](https://openreview.net/pdf?id=ryxnHhRqFm) [[code]](https://github.com/jasonwu0731/GLMP)
* **KB Retriever**: Entity-Consistent End-to-end Task-Oriented Dialogue System with KB Retriever. EMNLP-2019 [[pdf]](https://www.aclweb.org/anthology/D19-1013.pdf) [[data]](https://github.com/yizhen20133868/Retriever-Dialogue)
* **Mem2Seq**: Mem2Seq: Effectively Incorporating Knowledge Bases into End-to-End Task-Oriented Dialog Systems. ACL-2018 [[pdf]](https://www.aclweb.org/anthology/P18-1136.pdf) [[code]](https://github.com/HLTCHKUST/Mem2Seq)

## Transformer Series

* **GRET**: GRET: Global Representation Enhanced Transformer. AAAI-2020 [[pdf]](https://aaai.org/ojs/index.php/AAAI/article/view/6464)
* **Reformer**: REFORMER: THE EFFICIENT TRANSFORMER. ICLR-2020 [[pdf]](https://openreview.net/pdf?id=rkgNKkHtvB) [[code]](https://github.com/google/trax/tree/master/trax/models/reformer)
* **Transformer-XL**: Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context. ACL-2019 [[pdf]](https://www.aclweb.org/anthology/P19-1285.pdf) [[code]](https://github.com/kimiyoung/transformer-xl)
* **Universal Transformers**: UNIVERSAL TRANSFORMERS. ICLR-2019 [[pdf]](https://openreview.net/pdf?id=HyzdRiR9Y7) [[code]](https://github.com/tensorflow/tensor2tensor)
* **Transformer**: Attention Is All You Need. NeurIPS-2017 [[pdf]](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) [[code-official]](https://github.com/tensorflow/tensor2tensor) [[code-tf]](https://github.com/Kyubyong/transformer) [[code-py]](https://github.com/jadore801120/attention-is-all-you-need-pytorch)