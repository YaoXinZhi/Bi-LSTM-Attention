# --Bi-LSTM-Attention
深度学习课程作业-Bi-LSTM + Attention

paper: Zhou, Peng, et al. "Attention-based bidirectional long short-term memory networks for relation classification." Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers). 2016.


用了两个数据集 
短文本数据集 问题分类: https://github.com/u784799i/biLSTM_attn/tree/master/data
AG_corpus 新闻主题分类: http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html

查看帮助文档
python main_attention_lstm.py --help 


用 AG 数据集
python main_attention_lstm.py --dataset ag

用 问题分类数据集
python main_attention_lstm.py --dataset ques

如果想要存下 loss 和 acc 用于结果可视化
python main_attention_lstm.py --dataset ag --save_loss_acc

结果可视化代码也在了 直接修改注释的行 然后run就行。

如果要用改代码 点一个stat 
Merci ！
