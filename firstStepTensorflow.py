# 学习基本的 TensorFlow 概念
# 在 TensorFlow 中使用 LinearRegressor 类并基于单个输入特征预测各城市街区的房屋价值中位数
# 使用均方根误差 (RMSE) 评估模型预测的准确率
# 通过调整模型的超参数提高模型准确率

# 设置
import math
from IPython import display # 展示pandas数据容器
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

# 加载数据集
tf.logging.set_verbosity( tf.logging.ERROR )
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv( 'C:/Users/tuchao1996/Desktop/california_housing_train.csv', sep=',' )

california_housing_dataframe = california_housing_dataframe.reindex( np.random.permutation(california_housing_dataframe.index) )

def preprocess_features(california_housing_dataframe):
  """Prepares input features from California housing data set.

  Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
  """
  selected_features = california_housing_dataframe[
    ["latitude",
     "longitude",
     "housing_median_age",
     "total_rooms",
     "total_bedrooms",
     "population",
     "households",
     "median_income"]]
  processed_features = selected_features.copy()
  # Create a synthetic feature.
  processed_features["rooms_per_person"] = (
    california_housing_dataframe["total_rooms"] /
    california_housing_dataframe["population"])
  return processed_features

def preprocess_targets(california_housing_dataframe):
    """Prepares target features (i.e., labels) from California housing data set.

    Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
        from the California housing data set.
    Returns:
    A DataFrame that contains the target feature.
    """
    output_targets = pd.DataFrame()
    # Scale the target to be in units of thousands of dollars.
    output_targets["median_house_value"] = (
    california_housing_dataframe["median_house_value"] / 1000.0)
    return output_targets

def construct_feature_columns(input_features):
    return set( [tf.feature_column.numeric_column(my_feature) for my_feature in input_features ] )

def my_input_fn( features, targets, batch_size=1, shuffle=True, num_epochs=None ):
    features = { key:np.array(value) for key,value in dict(features).items() }
    
    ds = Dataset.from_tensor_slices( (features, targets) )
    ds = ds.batch(batch_size).repeat( num_epochs )
    
    if shuffle:
        ds = ds.shuffle( buffer_size=10000 )
    
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def train_model( learning_rate, steps, batch_size, training_examples, training_targets, validation_examples, validation_targets ):
    periods = 2
    steps_per_period = steps / periods

    # Create a linear regressor object.
    my_optimizer = tf.train.GradientDescentOptimizer( learning_rate=learning_rate )
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm( my_optimizer, 5.0 )
    linear_regressor = tf.estimator.LinearRegressor( feature_columns=construct_feature_columns(training_examples), optimizer=my_optimizer )

    # Create input functions
    display.display( training_examples )
    training_input_fn = lambda: my_input_fn( training_examples, training_targets["median_house_value"], batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn( training_examples, training_targets["median_house_value"], num_epochs=1, shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn( validation_examples, validation_targets["median_house_value"], num_epochs=1, shuffle=False )

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print( "Training model..." )
    print( "RMSE (on training data):" )
    training_rmse = []
    validation_rmse = []
    for period in range (0, periods):
        # Train the model, starting from the prior state.
        linear_regressor.train( input_fn=training_input_fn, steps=steps_per_period )
        # Take a break and compute predictions.
        training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        validation_predictions = linear_regressor.predict( input_fn=predict_validation_input_fn )
        validation_predictions = np.array( [item['predictions'][0] for item in validation_predictions] )

        # Compute training and validation loss.
        training_root_mean_squared_error = math.sqrt( metrics.mean_squared_error(training_predictions, training_targets) )
        validation_root_mean_squared_error = math.sqrt( metrics.mean_squared_error(validation_predictions, validation_targets) )
        # Occasionally print the current loss.
        print( "  period %02d : %0.2f" % (period, training_root_mean_squared_error) )
        # Add the loss metrics from this period to our list.
        training_rmse.append( training_root_mean_squared_error )
        validation_rmse.append( validation_root_mean_squared_error )
    print( "Model training finished." )


    # Output a graph of loss metrics over periods.
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()
    plt.xlabel("Periods")
    plt.ylabel("RMSE")
    plt.show()

    return linear_regressor

def fitness():
    # 构建第一个模型
    my_feature = california_housing_dataframe[['total_rooms']]
    feature_columns = [ tf.feature_column.numeric_column( 'total_rooms' ) ] # 特征列

    # 定义目标
    targets = california_housing_dataframe[ 'median_house_value' ]

    # 配置LinearRegressor
    my_optimizer = tf.train.FtrlOptimizer(learning_rate=0.000001)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm( my_optimizer, 5.0 )# 梯度裁剪应用到优化器

    linear_regressor = tf.estimator.LinearRegressor( feature_columns = feature_columns, optimizer=my_optimizer )

    # 训练模型
    _ = linear_regressor.train( input_fn=lambda:my_input_fn( my_feature, targets ), steps=100)

    # 评估模型
    prediction_input_fn = lambda:my_input_fn( my_feature, targets, num_epochs=1, shuffle=False )
    
    predictions = linear_regressor.predict( input_fn=prediction_input_fn )

    predictions = np.array( [ item['predictions'][0] for item in predictions ] )

    mean_squared_error = metrics.mean_squared_error( predictions, targets )
    root_mean_squared_error = math.sqrt( mean_squared_error )
    print( 'Mean squared error : %s' % mean_squared_error )
    print( 'Root squared error : %s' % root_mean_squared_error )

    # 绘图
    sample = california_housing_dataframe.sample( n=300 )
    
    x_0 = sample[ 'total_rooms' ].min()
    x_1 = sample[ 'total_rooms' ].max()

    weights = linear_regressor.get_variable_value( 'linear/linear_model/total_rooms/weights' )[0]
    bias = linear_regressor.get_variable_value( 'linear/linear_model/bias_weights' )

    y_0 = weights * x_0 + bias
    y_1 = weights * x_1 + bias

    plt.plot( [x_0, x_1], [y_0, y_1], c='r' )
    plt.scatter( sample['total_rooms'], sample['median_house_value'] )

    plt.show()

# def notation()

if __name__ == '__main__':
    # fitness()

    # Choose the first 12000 (out of 17000) examples for training.
    training_examples = preprocess_features(california_housing_dataframe.head(12000))
    training_targets = preprocess_targets(california_housing_dataframe.head(12000))

    # Choose the last 5000 (out of 17000) examples for validation.
    validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
    validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))

    # Double-check that we've done the right thing.
    print( "Training examples summary:" )
    display.display(training_examples.describe())
    print( "Validation examples summary:" )
    display.display(validation_examples.describe())

    print( "Training targets summary:" )
    display.display(training_targets.describe())
    print( "Validation targets summary:" )
    display.display(validation_targets.describe())

    correlation_dataframe = training_examples.copy()
    correlation_dataframe['target'] = training_targets['median_house_value']
    x = correlation_dataframe.corr()
    minimal_features = [ 'median_income', 'latitude', 'longitude' ]
    
    assert minimal_features, 'You must select at least one feature!'
    
    minimal_training_examples = training_examples[minimal_features] 
    minimal_validation_examples = validation_examples[minimal_features]

    train_model( learning_rate=1, steps=500, batch_size=100,training_examples=minimal_training_examples, training_targets=training_targets, validation_examples=minimal_validation_examples, validation_targets=validation_targets )

