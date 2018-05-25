import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn import cross_validation,metrics
import seaborn as sns
import MySQLdb as mdb
import MySQLdb.cursors as cursors
from datetime import datetime
import json

#conn = mdb.connect(host='10.251.193.133', user='bixiang', passwd='bixiang', db='xiujinniu', port=3306, charset='utf8',cursorclass=cursors.SSCursor)
#cursor = conn.cursor()

def predict_y():
    df = pd.read_csv('traindata_full.csv')

    # df.info()
    # df_obj = df.select_dtypes(include=['object']).copy()
    # print(df_obj.describe())


    # df['data_time'] = pd.to_datetime(df['data_time'])
    base_df = df.sort_values(['data_time'])
    # print(base_df)
    # print(base_df[['price_changerate_1']].apply(lambda x: abs(x)).describe())

    # base_df['dow'] = base_df['data_time'].apply(lambda x: x.dayofweek)
    # base_df['doy'] = base_df['time'].apply(lambda x: x.dayofyear)) & (base_df.data_time >= '2010-01-04')]
    # df_test = base_df.tail(1)
    # base_df['day'] = base_df['time'].apply(lambda x: x.day)
    # base_df['month'] = base_df['time'].apply(lambda x: x.month)
    # base_df['year'] = base_df['time'].apply(lambda x: x.year)


    df_train = base_df[(base_df.data_time <= '2016-12-31') & (base_df.data_time >= '2009-04-17')]
    # df_test = base_df.tail(1)
    df_test = base_df[(base_df.data_time >= '2017-01-01') & (base_df.data_time <= '2018-05-17') ] #19,23, 25
    # print(df_test)

    test_data_date = df_test['data_time']
    exc_cols = ['data_time', 'label_1', 'label_3', 'label_5', 'price_changerate_1_label','price_changerate_3_label','price_changerate_5_label']
    cols = [c for c in base_df.columns if c not in exc_cols]
    # print(cols)

    df_train[base_df.columns] = df_train[base_df.columns].fillna(0)
    df_test[base_df.columns] = df_test[base_df.columns].fillna(0)

    # df_train.to_csv('bixiang.csv')

    print(df_test.ix[:, cols].values)


    target_train = df_train['price_changerate_5_label'].values
    target_test = df_test['price_changerate_5_label'].values
    print(target_test)

    dtrain = xgb.DMatrix(df_train.ix[:, cols].values, label=target_train, feature_names=cols)
    dtest = xgb.DMatrix(df_test.ix[:, cols].values, label=target_test, feature_names=cols)


    # 设置数据集形式
    evallist = [(dtrain, 'train'),(dtest, 'test')]
    # evallist = [(dtrain, 'train')]

    param = {'booster': 'gblinear',
              'silent': 0,
              'eta': 0.1,  # 学习率，cv可以确定在该学习率下最佳的决策树数量
              'min_child_weight': 1,  # 默认1，决定最小叶子节点样本权重和。值过高,一个节点中会有太多的样本，进而导致欠拟合
              'max_depth': 6,  # 默认6，起始值在4-6之间比较合适
              'gamma': 0.3,  # 默认0，节点分裂所需的最小损失函数下降值。需要调整，起始可以选 0.1 0.2，越大越不容易过拟合
              'subsample': 0.8,  # 默认1，对于每棵树，随机采样的行占比， 起始值0.8较常见
              'colsample_bytree': 0.9,  # 默认1，对于每棵树。随机采样的列占比，起始值0.8较常见
              # 'lambda':1,                    #默认1  权重的L2正则化项,很少用到
              # 'alpha':0,                     #默认0, 权重的L1正则化项，应用在高纬度的情况，可以使算法速度更快
              # 'scale_pos_weight': 1,         #默认1，在各类别样本十分不平衡时，参数设定为一个正值，可以使算法更快收敛
              'objective': 'reg:linear',  # multi:softmax 多分类，返回预测的类别 (需要再设置一个 num_class)
              # 'num_class': 3,
              'eval_metric': 'rmse',  # logloss:负对数似然函数值    error:二分类错误率   auc:曲线下面积
              'seed': 55}  # 使用它可以复现随机数据的结果
    num_boost_round = 12000


    # cv 计算当前学习率下的最佳迭代次数
    # num_boost_round = 1200
    # if True:
    #     print('----------------- beginTrainCV --------------------')
    #     cvresult = xgb.cv(param, dtrain,
    #                            num_boost_round=num_boost_round,  # 也就是树量
    #                            nfold=10,
    #                            metrics='error',
    #                            early_stopping_rounds=100)
    #     num_boost_round = cvresult.shape[0]
    #
    # print('num_boost_round:' + str(num_boost_round))
    #
    # exit()

    # 训练
    print('----------------- beginTrain --------------------')
    history = xgb.train(param, dtrain, num_boost_round=num_boost_round, evals=evallist, early_stopping_rounds=100)
    #
    # 验证训练集的准确度
    # xgb_X_dtrain = xgb.DMatrix(df_train.ix[:, cols].values, missing=-999)
    result = history.predict(dtest)
    # print(type(model))
    print(result.shape)
    # print("Accuracy : %.4g" % metrics.accuracy_score(target_train, dtrain_predictions))


    fenmu = 0
    fenzi = 0
    for i in range(result.shape[0]):
        print(result[i],target_test[i])
        # with open ('result.csv','a') as f:
        #     f.write(str(result[i])+'\n')
        if abs(result[i]) > 0.02:
            fenmu = fenmu + 1
            if np.sign(target_test[i]) == np.sign(result[i]):
                fenzi = fenzi + 1

    print(fenzi, fenmu, fenzi / fenmu)

    # 保存模型
    history.save_model('xgboost_regressor.model')
    print('model has been saved')

    # model = xgb.XGBClassifier(n_estimators=152, max_depth=3,
    #                             learning_rate=0.1, subsample=0.8, colsample_bytree=0.3,scale_pos_weight=1,
    #                              silent=True, nthread=-1, seed=0, missing=None,objective='binary:logistic',
    #                              reg_alpha=1, reg_lambda=1,
    #                              gamma=0.3, min_child_weight=1,
    #                              max_delta_step=0,base_score=0.5)
    #
    #
    # model.fit(data_train, target_train, eval_metric='auc', eval_set=[(data_train, target_train),(data_test, target_test)],early_stopping_rounds=150)
    # print(model.get_fscore())

    # retrieve performance metrics
    # results = history.evals_result()
    # epochs = len(results['validation_0']['error'])
    # x_axis = range(0, epochs)
    # # plot log loss
    # fig, ax = pyplot.subplots()
    # ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    # ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
    # ax.legend()
    # pyplot.ylabel('Log Loss')
    # pyplot.title('XGBoost Log Loss')
    # pyplot.show()
    # # plot classification error
    # fig, ax = pyplot.subplots()
    # ax.plot(x_axis, results['validation_0']['error'], label='Train')
    # ax.plot(x_axis, results['validation_1']['error'], label='Test')
    # ax.legend()
    # pyplot.ylabel('Classification Error')
    # pyplot.title('XGBoost Classification Error')
    # pyplot.show()



    # print(model.predict_proba(data_test)[:,2])  # zhang
    # print(model.predict_proba(data_test)[:,1])  # ping
    # print(model.predict_proba(data_test)[:,0])  # die

    # count = 1
    # result = model.predict(data_test)
    # for i in range(len(data_test)):
    #     if list(target_test)[i] == result[i]:
    #         count = count + 1
    #     # print(list(target_test)[i], result[i])
    # print(count/len(data_test))

    #check train_data accuracy
    # print('The accuracy of train_data is ', model.score(data_train, target_train))
    # check test_data accuracy
    # print('The accuracy of test_data is ', model.score(data_test, target_test))

    # print(model.evals_result())

    # features = cols
    # mapFeat = dict(zip(['f'+str(i) for i in range(len(features))],features))
    # print(mapFeat)
    # print(model)
    # ts = pd.Series(model.booster().get_fscore())
    # ts.index = ts.reset_index()['index'].map(mapFeat)
    # ts.order()[-15:].plot(kind='barh', title=('feature importance'))

    # xgb.plot_importance(model,height=0.2,title='feature importance',xlabel='acc',ylabel='features', importance_type='weight',max_num_features=50,show_values=True)
    # xgb.plot_tree(model)
    # pyplot.show()

    # result_string = {"n": '%.3f' % float(model.predict_proba(data_test)[-1, 0]), "p": '%.3f' % float(model.predict_proba(data_test)[-1, 2]), "z": '%.3f' % float(model.predict_proba(data_test)[-1, 1])}
    # print(result_string)
    # result_json = json.dumps(result_string)
    # print(list(test_data_date)[0])
    # predict_date = list(pd.read_csv('date.csv')['day'])[list(pd.read_csv('date.csv')['day']).index(list(test_data_date)[0]) + 1]
    # print(predict_date)


    # now = datetime.now()
    # create_time = now.strftime('%Y-%m-%d %H:%M:%S')
    # print("""Insert into prediction_completed_data(target_cid, result_json, predict_date, create_time) values (%s,%s,%s,%s)""" % (label, repr(result_json), repr(predict_date), repr(create_time)))
    # cursor.execute("""Insert into prediction_completed_data(target_cid, result_json, predict_date, create_time) values (%s,%s,%s,%s)""" % (label, repr(result_json), repr(predict_date), repr(create_time)))
    # conn.commit()
    #
    # print("""select name from industry_data_catalog where catalog_id=%s"""% (label))
    # cursor.execute("""select name from industry_data_catalog where catalog_id=%s"""% (label))
    # rows = cursor.fetchall()
    # if rows:
    #     target_name = repr(rows[0][0])
    # else:
    #     target_name = '000'
    # print("""Insert into relation_prediction_target_explanatory(target_cid, explanatory_cid, create_time, explanatory_name, target_name) values (%s,%s,%s,%s,%s)""" % (label, '000', repr(create_time),'000',target_name))
    # cursor.execute("""Insert into relation_prediction_target_explanatory(target_cid, explanatory_cid, create_time, explanatory_name, target_name) values (%s,%s,%s,%s,%s)""" % (label, '000', repr(create_time),'000',target_name))
    # conn.commit()

predict_y()

