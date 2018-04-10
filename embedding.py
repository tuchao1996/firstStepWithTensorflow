#!/usr/bin/python
# -*- coding: utf-8 -*-

# 对影评数据进行情感分析的神经网络
# 训练一个情感分析模型，以预测某条评价总体是好评（1），还是差评（0）
# 字符串值term转换为特征矢量
# 嵌套理论知识：
# https://developers.google.cn/machine-learning/crash-course/embeddings/obtaining-embeddings

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 设置
import collections
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython import display
from sklearn import metrics

# tf.keras 包含文件下载和缓冲工具，用来检索数据集
tf.logging.set_verbosity(tf.logging.ERROR)
train_url = 'https://storage.googleapis.com/mledu-datasets/sparse-data-embedding/train.tfrecord'
train_path = tf.keras.utils.get_file(train_url.split('/')[-1], train_url)
test_url = 'https://storage.googleapis.com/mledu-datasets/sparse-data-embedding/test.tfrecord'
test_path = tf.keras.utils.get_file(test_url.split('/')[-1], test_url)

# 构建输入管道
def _parse_function( record ):
    '''Extract features and labels
    '''
    features = {
        'terms' : tf.VarLenFeature( dtype=tf.string ),
        'labels' : tf.FixedLenFeature( shape=[1], dtype=tf.float32 )
    }
    parsed_features = tf.parse_single_example( record, features )
    terms = parsed_features['terms'].values
    labels = parsed_features['labels']
    return {'terms':terms}, labels

def _input_fn( input_filenames, num_epochs=None, shuffle=True ):
    ds = tf.data.TFRecordDataset( input_filenames )
    ds = ds.map( _parse_function )

    if shuffle:
        ds = ds.shuffle( 10000 )

    ds = ds.padded_batch( 25, ds.output_shapes )

    ds = ds.repeat( num_epochs )

    features, labels = ds.make_one_shot_iterator().get_next()

    return features, labels

def modelDNN( feature_columns, train_path, test_path ):
    my_optimizer = tf.train.AdagradDAOptimizer( learning_rate=0.1, global_step=np.int64(1000) )
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm( my_optimizer, 5.0 )

    classifier = tf.estimator.DNNClassifier( 
        feature_columns=feature_columns,
        hidden_units=[ 10, 10 ],
        optimizer=my_optimizer 
    )
    classifier.train( 
        input_fn=lambda:_input_fn([train_path]), 
        steps=1000 
    )

    evaluation_metrics = classifier.evaluate( 
        input_fn=lambda:_input_fn([train_path]), 
        steps=1000 
    )
    print( 'Training set metrics:' )
    for m in evaluation_metrics:
        print( m, evaluation_metrics[m] )
    print( '---'*20 )

    evaluation_metrics = classifier.evaluate( input_fn=lambda:_input_fn([test_path]), steps=1000 )
    print( 'Test set metrics:' )
    for m in evaluation_metrics:
        print( m, evaluation_metrics[m] )
    print( '---'*20 )

def trainEmbeddingDNN( train_path, test_path ):
    f = open( 'C:/Users/tuchao1996/Desktop/terms.txt', 'r', encoding='utf8' )
    informative_terms = list( set(f.read().split()) )

    terms_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(key="terms", vocabulary_list=informative_terms)
    terms_embedding_column = tf.feature_column.embedding_column( terms_feature_column, dimension=2 )
    feature_columns = [terms_embedding_column]

    modelDNN( feature_columns, train_path, test_path )

if __name__ == '__main__' :
    start_time = time.time()
    print( time.strftime('%Y-%m-%d %X')+' Program has been launched!' )
    print( '---'*30 )

    # informative_terms = ("bad", "great", "best", "worst", "fun", "beautiful",
    #                     "excellent", "poor", "boring", "awful", "terrible",
    #                     "definitely", "perfect", "liked", "worse", "waste",
    #                     "entertaining", "loved", "unfortunately", "amazing",
    #                     "enjoyed", "favorite", "horrible", "brilliant", "highly",
    #                     "simple", "annoying", "today", "hilarious", "enjoyable",
    #                     "dull", "fantastic", "poorly", "fails", "disappointing",
    #                     "disappointment", "not", "him", "her", "good", "time",
    #                     "?", ".", "!", "movie", "film", "action", "comedy",
    #                     "drama", "family", "man", "woman", "boy", "girl")

    # 任务六：改进参数
    # informative_terms = None
    # f = open( 'C:/Users/tuchao1996/Desktop/terms.txt', 'r', encoding='utf8' )
    # informative_terms = list( set(f.read().split()) )

    # terms_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(key="terms", vocabulary_list=informative_terms)

    # 优化器
    # my_optimizer = tf.train.Ada gradDAOptimizer( learning_rate=0.1, global_step=np.int64(1000) )
    # my_optimizer = tf.contrib.estimator.clip_gradients_by_norm( my_optimizer, 5.0 )

    # # 任务一：使用具有稀疏输入和显示词汇表的线性模型
    # feature_columns = [ terms_feature_column ]
    # classifier = tf.estimator.LinearClassifier( 
    #     feature_columns=feature_columns,
    #     optimizer=my_optimizer
    # )
    # classifier.train( input_fn=lambda:_input_fn([train_path]), steps=1000 )
    # evaluation_metrics = classifier.evaluate( input_fn=lambda:_input_fn([train_path]), steps=1000 )
    
    # print( 'Training set metrics:' )
    # for m in evaluation_metrics:
    #     print( m, evaluation_metrics[m] )
    # print( '---'*20 )

    # evaluation_metrics = classifier.evaluate( input_fn=lambda:_input_fn([test_path]), steps=1000 )

    # print( 'Test set metrics:' )
    # for m in evaluation_metrics:
    #     print( m, evaluation_metrics[m] )
    # print( '---'*20 )

    # 任务二：使用深度神经网络（DNN）模型
    # feature_columns = [tf.feature_column.indicator_column(terms_feature_column)]
    # classifier = tf.estimator.DNNClassifier( 
    #     feature_columns=feature_columns,
    #     hidden_units=[ 20, 20 ],
    #     optimizer=my_optimizer 
    # )
    # try:
    #     classifier.train( input_fn=lambda:_input_fn([train_path]), steps=1000 )
    #     evaluation_metrics = classifier.evaluate( input_fn=lambda:_input_fn([train_path]), steps=1 )
    #     print( 'Training set metrics:' )
    #     for m in evaluation_metrics:
    #         print( m, evaluation_metrics[m] )
    #     print( '---'*20 )

    #     evaluation_metrics = classifier.evaluate( input_fn=lambda:_input_fn([test_path]), steps=1 )

    #     print( 'Test set metrics:' )
    #     for m in evaluation_metrics:
    #         print( m, evaluation_metrics[m] )
    #     print( '---'*20 )

    # except ValueError as err :
    #     print( err )

    # 任务三：在DNN中使用嵌入
    # 嵌入列会将稀疏数据作为输入，返回一个低纬度密集矢量作为输出
    # terms_embedding_column = tf.feature_column.embedding_column( terms_feature_column, dimension=2 )
    # feature_columns = [terms_embedding_column]

    # classifier = tf.estimator.DNNClassifier( 
    #     feature_columns=feature_columns,
    #     hidden_units=[ 10, 10 ],
    #     optimizer=my_optimizer 
    # )

    # try:
    #     classifier.train( input_fn=lambda:_input_fn([train_path]), steps=1000 )
    #     evaluation_metrics = classifier.evaluate( input_fn=lambda:_input_fn([train_path]), steps=1000 )
    #     print( 'Training set metrics:' )
    #     for m in evaluation_metrics:
    #         print( m, evaluation_metrics[m] )
    #     print( '---'*20 )

    #     evaluation_metrics = classifier.evaluate( input_fn=lambda:_input_fn([test_path]), steps=1000 )
    #     print( 'Test set metrics:' )
    #     for m in evaluation_metrics:
    #         print( m, evaluation_metrics[m] )
    #     print( '---'*20 )

    # except ValueError as err :
    #     print( err )

    # 任务四：确信模型中存在嵌入
    # for i in range( len(classifier.get_variable_names()) ):
    #     print( classifier.get_variable_names()[i] )
    # print( classifier.get_variable_value( 'dnn/input_from_feature_columns/input_layer/terms_embedding/embedding_weights' ).shape )

    # # 任务五：检查嵌入
    # embedding_matrix = classifier.get_variable_value( 'dnn/input_from_feature_columns/input_layer/terms_embedding/embedding_weights' )

    # for term_index in range( len(informative_terms) ):
    #     term_vector = np.zeros( len(informative_terms) )
    #     term_vector[term_index] = 1
    #     embedding_xy = np.matmul( term_vector, embedding_matrix )
    #     plt.text( embedding_xy[0], embedding_xy[1], informative_terms[term_index] )

    # plt.hold()
    # plt.plot( [-1.2*embedding_matrix.min(),1.2*embedding_matrix.max()], [0,0] )
    
    # plt.rcParams['figure.figsize'] = (12, 12)
    # plt.xlim( 1.2*embedding_matrix.min(), 1.2*embedding_matrix.max() )
    # plt.ylim( 1.2*embedding_matrix.min(), 1.2*embedding_matrix.max() )
    # plt.show()

    trainEmbeddingDNN(train_path, test_path)

    print( time.strftime('%Y-%m-%d %X')+' Program has been terminated!' )
    stop_time = time.time()
    spendTime = str( round( stop_time-start_time, 2 ) )
    print( 'Running time -> ' + spendTime + ' seconds.' )
