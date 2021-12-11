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
    - [Knowledge Grounded Dialogue](#knowledge-grounded-dialogue)
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

* **GLKS**: Thinking Globally, Acting Locally: Distantly Supervised Global-to-Local Knowledge Selection for Background Based Conversation. AAAI-2020 [[paper]](https://ojs.aaai.org//index.php/AAAI/article/view/6395) [[code]](https://github.com/PengjieRen/GLKS)
* **RefNet**: RefNet: A Reference-Aware Network for Background Based Conversation. AAAI-2020 [[paper]](https://ojs.aaai.org//index.php/AAAI/article/view/6370) [[code]](https://github.com/ChuanMeng/RefNet)
* **CaKe**: Improving Background Based Conversation with Context-aware Knowledge Pre-selection. arXiv-2019 [[paper]](https://arxiv.org/abs/1906.06685) [[code]](https://github.com/repozhang/bbc-pre-selection)

### Dataset for Dialogue

* **Pchatbot**: Pchatbot: A Large-Scale Dataset for Personalized Chatbot. SIGIR-2021 [[paper]](https://dl.acm.org/doi/10.1145/3404835.3463239) [[data]](https://github.com/qhjqhj00/SIGIR2021-Pchatbot)
* **MuTual**: A Dataset for Multi-Turn Dialogue Reasoning. ACL-2020 [[paper]](https://www.aclweb.org/anthology/2020.acl-main.130/) [[data]](https://github.com/Nealcly/MuTual)
* **Wizard of Wikipedia**: Wizard of Wikipedia: Knowledge-Powered Conversational Agents. ICLR-2019 [[paper]](https://openreview.net/forum?id=r1l73iRqKm) [[data]](https://parl.ai/projects/wizard_of_wikipedia/)
* **Holl-E**: Towards Exploiting Background Knowledge for Building Conversation Systems. EMNLP-2018 [[paper]](https://www.aclweb.org/anthology/D18-1255/) [[code]](https://github.com/nikitacs16/Holl-E)
* **OpenSubtitles2018**: OpenSubtitles2018: Statistical Rescoring of Sentence Alignments in Large, Noisy Parallel Corpora. LREC-2018 [[paper]](http://www.lrec-conf.org/proceedings/lrec2018/summaries/294.html) [[data]](https://opus.nlpl.eu/OpenSubtitles2018.php)
* **PERSONA-CHAT**: Personalizing Dialogue Agents: I have a dog, do you have pets too? ACL-2018 [[paper]](https://www.aclweb.org/anthology/P18-1205/)
* **DailyDialog**: DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset. IJCNLP-2017 [[paper]](https://www.aclweb.org/anthology/I17-1099/) [[data]](http://yanran.li/dailydialog)
* **Douban Conversation Corpus**: Sequential Matching Network: A New Architecture for Multi-turn Response Selection in Retrieval-Based Chatbots. ACL-2017 [[paper]](https://www.aclweb.org/anthology/P17-1046/) [[data]](https://github.com/MarkWuNLP/MultiTurnResponseSelection)
* **OpenSubtitles2016**: OpenSubtitles2016: Extracting Large Parallel Corpora from Movie and TV Subtitles. LREC-2016 [[paper]](http://www.lrec-conf.org/proceedings/lrec2016/summaries/947.html) [[data]](http://opus.lingfil.uu.se/OpenSubtitles2016.php)
* **Ubuntu Dialogue Corpus**: The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems. SIGDIAL-2015 [[paper]](https://www.aclweb.org/anthology/W15-4640/) [[data]](http://www.cs.mcgill.ca/~jpineau/datasets/ubuntu-corpus-1.0)
* **Cornell Movie Dialogs Corpus**: Chameleons in Imagined Conversations: A New Approach to Understanding Coordination of Linguistic Style in Dialogs. CMCL@ACL-2011 [[paper]](https://www.aclweb.org/anthology/W11-0609/) [[data]](https://www.cs.cornell.edu/~cristian/Chameleons_in_imagined_conversations.html)

### Emotional Dialogue

* **MOJITALK**: MOJITALK: Generating Emotional Responses at Scale. ACL-2018 [[paper]](https://www.aclweb.org/anthology/P18-1104/)

### Evaluation of Dialogue

* **Holistic Evaluation**: Towards Holistic and Automatic Evaluation of Open-Domain Dialogue Generation. ACL-2020 [[paper]](https://www.aclweb.org/anthology/2020.acl-main.333/) [[code]](https://github.com/alexzhou907/dialogue_evaluation)
* **RS**: Evaluating Dialogue Generation Systems via Response Selection. ACL-2020 [[paper]](https://www.aclweb.org/anthology/2020.acl-main.55/) [[test set]](https://github.com/cl-tohoku/eval-via-selection)
* **υBLEU**: υBLEU: Uncertainty-Aware Automatic Evaluation Method for Open-Domain Dialogue Systems. ACL-2020 [[paper]](https://www.aclweb.org/anthology/2020.acl-srw.27/) [[code&datasets]](http://www.tkl.iis.u-tokyo.ac.jp/~tsuta/acl-srw-2020/)
* **Empirical Study**: How NOT To Evaluate Your Dialogue System: An Empirical Study of Unsupervised Evaluation Metrics for Dialogue Response Generation. EMNLP-2016 [[paper]](https://www.aclweb.org/anthology/D16-1230/)
* **BLEU**: BLEU: a Method for Automatic Evaluation of Machine Translation. ACL-2002 [[paper]](https://www.aclweb.org/anthology/P02-1040/)

### Knowledge Grounded Dialogue

* **HADG**: Knowledge-aware Dialogue Generation with Hybrid Attention (Student Abstract). AAAI-2021 [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17972)
* **ConKADI**: Diverse and Informative Dialogue Generation with Context-Specific Commonsense Knowledge Awareness. ACL-2020 [[paper]](https://www.aclweb.org/anthology/2020.acl-main.515/) [[code]](https://github.com/pku-sixing/ACL2020-ConKADI)
* **Div-Non-Conv**: Diversifying Dialogue Generation with Non-Conversational Text. ACL-2020 [[paper]](https://www.aclweb.org/anthology/2020.acl-main.634/) [[code]](https://github.com/chin-gyou/Div-Non-Conv)
* **KIC**: Generating Informative Conversational Response using Recurrent Knowledge-Interaction and Knowledge-Copy. ACL-2020 [[paper]](https://www.aclweb.org/anthology/2020.acl-main.6/)
* **Low-Resource KG**: Low-Resource Knowledge-Grounded Dialogue Generation. ICLR-2020 [[paper]](https://openreview.net/forum?id=rJeIcTNtvS)
* **PIPM-KDBTS**: Bridging the Gap between Prior and Posterior Knowledge Selection for Knowledge-Grounded Dialogue Generation. EMNLP-2020 [[paper]](https://www.aclweb.org/anthology/2020.emnlp-main.275/) [[code]](https://github.com/youngornever/bridge_latent_knowledge_selection_gap_for_conversation)
* **TransDG**: Improving Knowledge-Aware Dialogue Generation via Knowledge Base Question Answering. AAAI-2020 [[paper]](https://ojs.aaai.org//index.php/AAAI/article/view/6453) [[code]](https://github.com/siat-nlp/TransDG)
* **ZRKGC**: Zero-Resource Knowledge-Grounded Dialogue Generation. NeurIPS-2020 [[paper]](https://proceedings.neurips.cc/paper/2020/hash/609c5e5089a9aa967232aba2a4d03114-Abstract.html) [[code]](https://github.com/nlpxucan/ZRKGC)
* **PostKS**: Learning to Select Knowledge for Response Generation in Dialog Systems. IJCAI-2019 [[paper]](https://www.ijcai.org/proceedings/2019/0706/) [[code]](https://github.com/bzantium/Posterior-Knowledge-Selection)
* **Mem2Seq**: Mem2Seq: Effectively Incorporating Knowledge Bases into End-to-End Task-Oriented Dialog Systems. ACL-2018 [[paper]](https://www.aclweb.org/anthology/P18-1136/) [[code]](https://github.com/HLTCHKUST/Mem2Seq)
* **NKD**: Knowledge Diffusion for Neural Dialogue Generation. ACL-2018 [[paper]](https://www.aclweb.org/anthology/P18-1138/) [[code]](https://github.com/liushuman/neural-knowledge-diffusion)

### Open-domain Dialogue

* **MDFN**: Filling the Gap of Utterance-aware and Speaker-aware Representation for Multi-turn Dialogue. AAAI-2021 [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17582) [[code]](https://github.com/comprehensiveMap/MDFN)
* **Adp Multi-CL**: Learning from Easy to Complex: Adaptive Multi-Curricula Learning for Neural Dialogue Generation. AAAI-2020 [[paper]](https://aaai.org/ojs/index.php/AAAI/article/view/6244)
* **DialogRPT**: Dialogue Response Ranking Training with Large-Scale Human Feedback Data. EMNLP-2020 [[paper]](https://www.aclweb.org/anthology/2020.emnlp-main.28/) [[code]](https://github.com/golsun/DialogRPT)
* **PS**: Prototype-to-Style: Dialogue Generation with Style-Aware Editing on Retrieval Memory. arXiv-2020 [[paper]](https://arxiv.org/abs/2004.02214)
* **Edit-N-Rerank**: Response Generation by Context-aware Prototype Editing. AAAI-2019 [[paper]](https://ojs.aaai.org//index.php/AAAI/article/view/4714) [[code]](https://github.com/MarkWuNLP/ResponseEdit)
* **SeqGen**: Retrieval-guided Dialogue Response Generation via a Matching-to-Generation Framework. EMNLP-2019 [[paper]](https://www.aclweb.org/anthology/D19-1195/) [[code]](https://github.com/jcyk/seqgen)
* **SR**: Skeleton-to-Response: Dialogue Generation Guided by Retrieval Memory. NAACL-2019 [[paper]](https://www.aclweb.org/anthology/N19-1124/)
* **NEXUS**: NEXUS Network: Connecting the Preceding and the Following in Dialogue Generation. EMNLP-2018 [[paper]](https://www.aclweb.org/anthology/D18-1463/)
* **CVAE**: Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders. ACL-2017 [[paper]](https://www.aclweb.org/anthology/P17-1061/)
* **DRL**: Deep Reinforcement Learning for Dialogue Generation. EMNLP-2016 [[paper]](https://www.aclweb.org/anthology/D16-1127/)
* **MMI**: A Diversity-Promoting Objective Function for Neural Conversation Models. NAACL-2016 [[paper]](https://www.aclweb.org/anthology/N16-1014/)
* **NCM**: A Neural Conversational Model. arXiv-2015 [[paper]](https://arxiv.org/abs/1506.05869)

### Personalized Dialogue

* **DHAP**: One Chatbot Per Person: Creating Personalized Chatbots based on Implicit User Profiles. SIGIR-2021 [[paper]](https://dl.acm.org/doi/10.1145/3404835.3462828) [[code]](https://github.com/zhengyima/DHAP)
* **FewShot-PCM**: Learning from My Friends: Few-Shot Personalized Conversation Systems via Social Networks. AAAI-2021 [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17638) [[code]](https://github.com/tianzhiliang/FewShotPersonaConvData)
* **GCC**: Large Scale Multi-Actor Generative Dialog Modeling. ACL-2020 [[paper]](https://www.aclweb.org/anthology/2020.acl-main.8/)
* **GDR**: Generate, Delete and Rewrite: A Three-Stage Framework for Improving Persona Consistency of Dialogue Generation. ACL-2020 [[paper]](https://www.aclweb.org/anthology/2020.acl-main.516/)
* **KvPI**: Proﬁle Consistency Identiﬁcation for Open-domain Dialogue Agents. EMNLP-2020 [[paper]](https://www.aclweb.org/anthology/2020.emnlp-main.539/) [[code]](https://github.com/songhaoyu/KvPI)
* **P<sup>2</sup> Bot**: You Impress Me: Dialogue Generation via Mutual Persona Perception. ACL-2020 [[paper]](https://www.aclweb.org/anthology/2020.acl-main.131/) [[code]](https://github.com/SivilTaram/Persona-Dialogue-Generation)
* **PAGenerator**: Guiding Variational Response Generator to Exploit Persona. ACL-2020 [[paper]](https://www.aclweb.org/anthology/2020.acl-main.7/)
* **PEE**: A Neural Topical Expansion Framework for Unstructured Persona-Oriented Dialogue Generation. ECAI-2020 [[paper]](https://ebooks.iospress.nl/publication/55146)
* **Persona-sparse**: A Pre-training Based Personalized Dialogue Generation Model with Persona-sparse Data. AAAI-2020 [[paper]](https://ojs.aaai.org//index.php/AAAI/article/view/6518)
* **RCDG**: Generating Persona Consistent Dialogues by Exploiting Natural Language Inference. AAAI-2020 [[paper]](https://ojs.aaai.org//index.php/AAAI/article/view/6417) [[code]](https://github.com/songhaoyu/RCDG)
* **Self-Consciousness**: Will I Sound Like Me? Improving Persona Consistency in Dialogues through Pragmatic Self-Consciousness. EMNLP-2020 [[paper]](https://www.aclweb.org/anthology/2020.emnlp-main.65/) [[code]](https://github.com/skywalker023/pragmatic-consistency)
* **PAML**: Personalizing Dialogue Agents via Meta-Learning. ACL-2019 [[paper]](https://www.aclweb.org/anthology/P19-1542/) [[code]](https://github.com/HLTCHKUST/PAML)
* **PerCVAE**: Exploiting Persona Information for Diverse Generation of Conversational Responses. IJCAI-2019 [[paper]](https://www.ijcai.org/Proceedings/2019/0721/) [[code]](https://github.com/vsharecodes/percvae)
* **PerDG**: Personalized Dialogue Generation with Diversified Traits. arXiv-2019 [[paper]](https://arxiv.org/abs/1901.09672)
* **NPRG-DM**: Neural personalized response generation as domain adaptation. WWW-2018 [[paper]](https://link.springer.com/article/10.1007%2Fs11280-018-0598-6)
* **PCCM**: Assigning Personality Profile to a Chatting Machine for Coherent Conversation Generation. IJCAI-2018 [[paper]](https://www.ijcai.org/Proceedings/2018/0595/)
* **PerDA**: Training Millions of Personalized Dialogue Agents. EMNLP-2018 [[paper]](https://www.aclweb.org/anthology/D18-1298/)
* **CoPerHED**: Exploring Personalized Neural Conversational Models. IJCAI-2017 [[paper]](https://www.ijcai.org/proceedings/2017/0521/)
* **PRG-DM**: Personalized Response Generation via Domain adaptation. SIGIR-2017 [[paper]](https://dl.acm.org/doi/10.1145/3077136.3080706)
* **PerNCM**: A Persona-Based Neural Conversation Model. ACL-2016 [[paper]](https://www.aclweb.org/anthology/P16-1094/)
* **PERSONAGE**: PERSONAGE: Personality Generation for Dialogue. ACL-2007 [[paper]](https://www.aclweb.org/anthology/P07-1063/)

### Survey on Dialogue

* **A Systematic Survey**: Recent Advances in Deep Learning Based Dialogue Systems: A Systematic Survey. arXiv-2021 [[paper]](https://arxiv.org/abs/2105.04387)
* **Challenges**: Challenges in Building Intelligent Open-domain Dialog Systems. TOIS-2020 [[paper]](https://dl.acm.org/doi/10.1145/3383123)
* **Recipes**: Recipes for building an open-domain chatbot. arXiv-2020 [[paper]](https://arxiv.org/abs/2004.13637)
* **Recent Advances and New Frontiers**: A Survey on Dialogue Systems: Recent Advances and New Frontiers. SIGKDD-2017 [[paper]](https://dl.acm.org/doi/10.1145/3166054.3166058)

### Task-oriented Dialogue

* **DAST**: A Student-Teacher Architecture for Dialog Domain Adaptation under the Meta-Learning Setting. AAAI-2021 [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17614)
* **DDMN**: Dual Dynamic Memory Network for End-to-End Multi-turn Task-oriented Dialog Systems. COLING-2020 [[paper]](https://www.aclweb.org/anthology/2020.coling-main.362/) [[code]](https://github.com/siat-nlp/DDMN)
* **DF-Net**: Dynamic Fusion Network for Multi-Domain End-to-end Task-Oriented Dialog. ACL-2020 [[paper]](https://www.aclweb.org/anthology/2020.acl-main.565/) [[code]](https://github.com/LooperXX/DF-Net)
* **MALA**: MALA: Cross-Domain Dialogue Generation with Action Learning. AAAI-2020 [[paper]](https://aaai.org/ojs/index.php/AAAI/article/view/6306)
* **BossNet**: Disentangling Language and Knowledge in Task-Oriented Dialogs. NAACL-2019 [[paper]](https://www.aclweb.org/anthology/N19-1126/) [[code]](https://github.com/dair-iitd/BossNet)
* **GLMP**: GLOBAL-TO-LOCAL MEMORY POINTER NETWORKS FOR TASK-ORIENTED DIALOGUE. ICLR-2019 [[paper]](https://openreview.net/forum?id=ryxnHhRqFm) [[code]](https://github.com/jasonwu0731/GLMP)
* **KB Retriever**: Entity-Consistent End-to-end Task-Oriented Dialogue System with KB Retriever. EMNLP-2019 [[paper]](https://www.aclweb.org/anthology/D19-1013/) [[data]](https://github.com/yizhen20133868/Retriever-Dialogue)
* **TRADE**: Transferable Multi-Domain State Generator for Task-Oriented Dialogue Systems. ACL-2019 [[paper]](https://www.aclweb.org/anthology/P19-1078/) [[code]](https://github.com/jasonwu0731/trade-dst)
* **DSR**: Sequence-to-Sequence Learning for Task-oriented Dialogue with Dialogue State Representation. COLING-2018 [[paper]](https://www.aclweb.org/anthology/C18-1320/)
* **StateNet**: Towards Universal Dialogue State Tracking. EMNLP-2018 [[paper]](https://www.aclweb.org/anthology/D18-1299/)
* **TSCP**: Sequicity: Simplifying Task-oriented Dialogue Systems with Single Sequence-to-Sequence Architectures. ACL-2018 [[paper]](https://www.aclweb.org/anthology/P18-1133/) [[code]](https://github.com/WING-NUS/sequicity)

## Pre-trained Language Model

* **ALBERT**: ALBERT: A Lite BERT for Self-supervised Learning of Language Representations. ICLR-2020 [[paper]](https://openreview.net/forum?id=H1eA7AEtvS) [[code]](https://github.com/google-research/ALBERT)
* **BERT**: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-2019 [[paper]](https://www.aclweb.org/anthology/N19-1423) [[code]](https://github.com/google-research/bert)
* **ERNIE**: ERNIE: Enhanced Language Representation with Informative Entities. ACL-2019 [[paper]](https://www.aclweb.org/anthology/P19-1139/) [[code]](https://github.com/thunlp/ERNIE)
* **Interpret_BERT**: What does BERT learn about the structure of language? ACL-2019 [[paper]](https://www.aclweb.org/anthology/P19-1356/) [[code]](https://github.com/ganeshjawahar/interpret_bert)
* **UniLM**: Unified Language Model Pre-training for Natural Language Understanding and Generation. NeurIPS-2019 [[paper]](https://proceedings.neurips.cc/paper/2019/hash/c20bb2d9a50d5ac1f713f8b34d9aac5a-Abstract.html) [[code]](https://github.com/microsoft/unilm)
* **XLNet**: XLNet: Generalized Autoregressive Pretraining for Language Understanding. NeurIPS-2019 [[paper]](https://proceedings.neurips.cc/paper/2019/hash/dc6a7e655d7e5840e66733e9ee67cc69-Abstract.html) [[code]](https://github.com/zihangdai/xlnet)

## Text Summarization

### Abstractive

* **GTTP**: Get To The Point: Summarization with Pointer-Generator Networks. ACL-2017 [[paper]](https://www.aclweb.org/anthology/P17-1099/) [[code]](https://github.com/abisee/pointer-generator)

## Transformer Series

* **GRET**: GRET: Global Representation Enhanced Transformer. AAAI-2020 [[paper]](https://aaai.org/ojs/index.php/AAAI/article/view/6464)
* **Reformer**: REFORMER: THE EFFICIENT TRANSFORMER. ICLR-2020 [[paper]](https://openreview.net/forum?id=rkgNKkHtvB) [[code]](https://github.com/google/trax/tree/master/trax/models/reformer)
* **Transformer-XL**: Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context. ACL-2019 [[paper]](https://www.aclweb.org/anthology/P19-1285/) [[code]](https://github.com/kimiyoung/transformer-xl)
* **Universal Transformers**: UNIVERSAL TRANSFORMERS. ICLR-2019 [[paper]](https://openreview.net/forum?id=HyzdRiR9Y7) [[code]](https://github.com/tensorflow/tensor2tensor)
* **Transformer**: Attention Is All You Need. NeurIPS-2017 [[paper]](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) [[code-official]](https://github.com/tensorflow/tensor2tensor) [[code-tf]](https://github.com/Kyubyong/transformer) [[code-py]](https://github.com/jadore801120/attention-is-all-you-need-pytorch)

## Word Embedding

* **Doc2Vec**: Distributed Representations of Sentences and Documents. ICML-2014 [[paper]](http://proceedings.mlr.press/v32/le14.html)
* **Glove**: GloVe: Global Vectors for Word Representation. EMNLP-2014 [[paper]](https://www.aclweb.org/anthology/D14-1162/) [[code]](https://github.com/stanfordnlp/GloVe)
* **Word2Vec-Extension**: Distributed Representations of Words and Phrases and their Compositionality. NeurIPS-2013 [[paper]](https://proceedings.neurips.cc/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html)
* **Word2Vec-Oringinal**: Efficient Estimation of Word Representations in Vector Space. ICLR Workshop Poster-2013 [[paper]](https://arxiv.org/abs/1301.3781) [[code]](https://code.google.com/p/word2vec/)