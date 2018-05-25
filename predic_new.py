import pandas as pd
import xgboost as xgb
import numpy as np
from matplotlib import pyplot
import seaborn as sns
import MySQLdb as mdb
import MySQLdb.cursors as cursors
from datetime import datetime
import json

#conn = mdb.connect(host='10.251.193.133', user='bixiang', passwd='bixiang', db='xiujinniu', port=3306, charset='utf8',cursorclass=cursors.SSCursor)
#cursor = conn.cursor()

def predict_y():
    df = pd.read_csv('traindata.csv')

    # df.info()
    # df_obj = df.select_dtypes(include=['object']).copy()
    # print(df_obj.describe())


    # df['data_time'] = pd.to_datetime(df['data_time'])
    base_df = df.sort_values(['data_time'])
    # print(base_df)

    # base_df['dow'] = base_df['data_time'].apply(lambda x: x.dayofweek)
    # base_df['doy'] = base_df['time'].apply(lambda x: x.dayofyear)) & (base_df.data_time >= '2010-01-04')]
    # df_test = base_df.tail(1)
    # base_df['day'] = base_df['time'].apply(lambda x: x.day)
    # base_df['month'] = base_df['time'].apply(lambda x: x.month)
    # base_df['year'] = base_df['time'].apply(lambda x: x.year)

    df_train = base_df[(base_df.data_time <= '2016-12-31') & (base_df.data_time >= '2009-04-17')]
    # df_test = base_df.tail(1)
    df_test = base_df[(base_df.data_time >= '2017-01-01') & (base_df.data_time <= '2018-04-19')] #19,23, 25
    # print(df_test)

    test_data_date = df_test['data_time']
    exc_cols = ['data_time', 'label_1', 'label_3', 'label_5', 'price', 'price_change_1', 'price_change_3', 'price_change_5','price_changerate_1', 'price_changerate_3', 'price_changerate_5']
    cols = [c for c in base_df.columns if c not in exc_cols]
    # print(cols)



    data_train = np.array(df_train.ix[:, cols])
    data_test = np.array(df_test.ix[:, cols])



    target_train = df_train['label_5'].values
    target_test = df_test['label_5'].values


    model = xgb.XGBClassifier(n_estimators=152, max_depth=6,
                                learning_rate=0.1, subsample=0.8, colsample_bytree=0.9,
                                 silent=True, nthread=-1, seed=55, missing=None,objective='binary:logistic',
                                 gamma=0.3, min_child_weight=1,
                                 max_delta_step=0,base_score=0.5)


    model.fit(data_train, target_train, eval_metric='error', eval_set=[(data_train, target_train),(data_test, target_test)],early_stopping_rounds=2)





    # print(model.predict_proba(data_test)[:,2])  # zhang
    # print(model.predict_proba(data_test)[:,1])  # ping
    # print(model.predict_proba(data_test)[:,0])  # die

    print(len(data_test))
    count = 1
    result = model.predict(data_test)
    for i in range(len(data_test)):
        if list(target_test)[i] == result[i]:
            count = count + 1
        # print(list(target_test)[i], result[i])
    print(count/len(data_test))

    #check train_data accuracy
    # print('The accuracy of train_data is ', model.score(data_train, target_train))
    # print('The accuracy of test_data is ', model.score(data_test, target_test))

    # print(model.evals_result())

    # features = cols
    # mapFeat = dict(zip(['f'+str(i) for i in range(len(features))],features))
    # print(mapFeat)
    # print(model)
    # ts = pd.Series(model.booster().get_fscore())
    # ts.index = ts.reset_index()['index'].map(mapFeat)
    # ts.order()[-15:].plot(kind='barh', title=('feature importance'))

    xgb.plot_importance(model,height=0.2,title='feature importance',xlabel='acc',ylabel='features', importance_type='weight',max_num_features=50,show_values=True)
    # xgb.plot_tree(model)
    pyplot.show()

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

