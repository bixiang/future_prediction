import os
import json
import pandas as pd
from copy import deepcopy
from datetime import datetime
from retrying import retry
# from handle_for_lstm import handle_for_lstm
import re
import os



def df2dic(df, keyname_piece):
    df = df.T.to_dict('records')
    rank_id = df[0]
    rank_volumn = df[1]
    rank_id_new = deepcopy(rank_id)
    rank_volumn_new = deepcopy(rank_volumn)
    for old_key in rank_id:
        new_key = str(old_key + 1) + '_' + keyname_piece + '_id'
        rank_id_new[new_key] = rank_id[old_key]
        del rank_id_new[old_key]
    # print(rank_id_new)
    for old_key in rank_volumn:
        new_key = str(old_key + 1) + '_' + keyname_piece + '_volumn'
        rank_volumn_new[new_key] = rank_volumn[old_key]
        del rank_volumn_new[old_key]
    # print(rank_volumn_new)
    rank_id_new.update(rank_volumn_new)
    # return rank_id_new
    return rank_volumn_new

def gen_rt_db(topn, datpath):
    '''
    function of handling source trading volume rank data got from website 
    :return: list of dict, every row of a dataframe 
    '''

    final_daydata = {}


    def func(x):
        return 0 if not x else x

    if '.xls' not in datpath:
        with open(datpath, 'r') as f:
            data_json = json.loads(f.read())
            data_time = re.compile('shfe(.*?).dat').findall(datpath)[0]
            data_body = data_json['o_cursor']
            final_daydata['data_time'] = datetime.strftime(pd.to_datetime(data_time), '%Y-%m-%d')
        base_df = pd.DataFrame(data_body)

        # base_df['INSTRUMENTID'] = base_df['INSTRUMENTID'].apply(lambda x: x.rstrip())
        # base_df['PARTICIPANTABBR1'] = base_df['PARTICIPANTABBR1'].apply(lambda x: x.strip())
        # base_df['PARTICIPANTABBR2'] = base_df['PARTICIPANTABBR2'].apply(lambda x: x.strip())
        # base_df['PARTICIPANTABBR3'] = base_df['PARTICIPANTABBR3'].apply(lambda x: x.strip())
        base_df[['PARTICIPANTABBR1', 'PARTICIPANTABBR2', 'PARTICIPANTABBR3', 'INSTRUMENTID']] = base_df[['PARTICIPANTABBR1', 'PARTICIPANTABBR2', 'PARTICIPANTABBR3','INSTRUMENTID']].applymap(lambda x: x.strip())
        # base_df['CJ1'] = base_df['CJ1'].apply(lambda x: func(x))
        # base_df['CJ2'] = base_df['CJ2'].apply(lambda x: func(x))
        # base_df['CJ3'] = base_df['CJ3'].apply(lambda x: func(x))
        # base_df['CJ1_CHG'] = base_df['CJ1_CHG'].apply(lambda x: func(x))
        # base_df['CJ2_CHG'] = base_df['CJ2_CHG'].apply(lambda x: func(x))
        # base_df['CJ3_CHG'] = base_df['CJ3_CHG'].apply(lambda x: func(x))
        base_df[['CJ1', 'CJ1_CHG', 'CJ2', 'CJ2_CHG','CJ3', 'CJ3_CHG']] = base_df[['CJ1', 'CJ1_CHG', 'CJ2', 'CJ2_CHG','CJ3', 'CJ3_CHG']].applymap(lambda x: func(x))

        # 合计
        # sum_dic = base_df[(base_df.RANK == 999) & (base_df.INSTRUMENTID.str.contains('rb'))].agg(
        #     {'CJ1': 'sum', 'CJ1_CHG': 'sum', 'CJ2': 'sum', 'CJ2_CHG': 'sum', 'CJ3': 'sum', 'CJ3_CHG': 'sum'}).to_dict()
        #
        # final_daydata.update(sum_dic)


        df = base_df[(base_df.INSTRUMENTID.str.contains('rb')) &
                     (base_df.PARTICIPANTABBR1) & (~base_df.PARTICIPANTABBR1.str.contains('期货公司')) &
                     (base_df.PARTICIPANTABBR2) & (~base_df.PARTICIPANTABBR2.str.contains('期货公司')) &
                     (base_df.PARTICIPANTABBR3) & (~base_df.PARTICIPANTABBR3.str.contains('期货公司'))]

    else:
        data_time = re.compile('dafe(.*?).xls').findall(datpath)[0]
        final_daydata['data_time'] = datetime.strftime(pd.to_datetime(data_time), '%Y-%m-%d')
        df = pd.read_excel(datpath, skiprows=[0, 1, 2, 24], usecols=['名次', '会员简称', '成交量', '增减', '会员简称.1', '持买单量', '增减.1', '会员简称.2', '持卖单量', '增减.2'])
        df.columns = ['RANK', 'PARTICIPANTABBR1', 'CJ1', 'CJ1_CHG', 'PARTICIPANTABBR2', 'CJ2', 'CJ2_CHG', 'PARTICIPANTABBR3', 'CJ3', 'CJ3_CHG']  # INSTRUMENTID
        # df['CJ1'] = df['CJ1'].apply(lambda x: x.replace(',',''))
        # df['CJ2'] = df['CJ2'].apply(lambda x: x.replace(',',''))
        # df['CJ3'] = df['CJ3'].apply(lambda x: x.replace(',',''))
        # df['CJ1_CHG'] = df['CJ1_CHG'].apply(lambda x: x.replace(',',''))
        # df['CJ2_CHG'] = df['CJ2_CHG'].apply(lambda x: x.replace(',',''))
        # df['CJ3_CHG'] = df['CJ3_CHG'].apply(lambda x: x.replace(',',''))
        if df.shape[0] != 0:
            df[['CJ1', 'CJ1_CHG', 'CJ2', 'CJ2_CHG','CJ3', 'CJ3_CHG']] = df[['CJ1', 'CJ1_CHG', 'CJ2', 'CJ2_CHG','CJ3', 'CJ3_CHG']].applymap(lambda x: str(x).replace(',','')).apply(pd.to_numeric)


    bix = df.groupby(['PARTICIPANTABBR1', 'PARTICIPANTABBR2', 'PARTICIPANTABBR3']).agg({'CJ1': 'sum', 'CJ2': 'sum', 'CJ3': 'sum'})
    print(bix)

    # 前n名的量和公司id
    CJ1top = df.groupby('PARTICIPANTABBR1', as_index=False).agg({'CJ1':'sum'}).sort_values(ascending=False, by='CJ1').head(topn).reset_index(drop=True)
    CJ1_CHGtop = df.groupby('PARTICIPANTABBR1', as_index=False).agg({'CJ1_CHG':'sum'}).sort_values(ascending=False, by='CJ1_CHG').head(topn).reset_index(drop=True)
    CJ2top = df.groupby('PARTICIPANTABBR2', as_index=False).agg({'CJ2':'sum'}).sort_values(ascending=False, by='CJ2').head(topn).reset_index(drop=True)
    CJ2_CHGtop = df.groupby('PARTICIPANTABBR2', as_index=False).agg({'CJ2_CHG':'sum'}).sort_values(ascending=False, by='CJ2_CHG').head(topn).reset_index(drop=True)
    CJ3top = df.groupby('PARTICIPANTABBR3', as_index=False).agg({'CJ3':'sum'}).sort_values(ascending=False, by='CJ3').head(topn).reset_index(drop=True)
    CJ3_CHGtop = df.groupby('PARTICIPANTABBR3', as_index=False).agg({'CJ3_CHG':'sum'}).sort_values(ascending=False, by='CJ3_CHG').head(topn).reset_index(drop=True)

    # print(CJ1_CHGtop.dtypes)
    # print(CJ1_CHGtop)



    CJ1top_dic = df2dic(CJ1top, 'CJ1top')
    CJ1_CHGtop_dic = df2dic(CJ1_CHGtop, 'CJ1_CHGtop')
    CJ2top_dic = df2dic(CJ2top, 'CJ2top')
    CJ2_CHGtop_dic = df2dic(CJ2_CHGtop, 'CJ2_CHGtop')
    CJ3top_dic = df2dic(CJ3top, 'CJ3top')
    CJ3_CHGtop_dic = df2dic(CJ3_CHGtop, 'CJ3_CHGtop')



    final_daydata.update(CJ1top_dic)
    final_daydata.update(CJ1_CHGtop_dic)
    final_daydata.update(CJ2top_dic)
    final_daydata.update(CJ2_CHGtop_dic)
    final_daydata.update(CJ3top_dic)
    final_daydata.update(CJ3_CHGtop_dic)


    return final_daydata

@retry(stop_max_attempt_number=3)
def download_sh(date, procuct):
    import urllib.request
    url = 'http://www.shfe.com.cn/data/dailydata/kx/pm{}.dat'.format(date)
    urllib.request.urlretrieve(url, os.path.join(os.path.abspath(os.path.dirname(__file__)), '{}_datdata/shfe{}.dat'.format(procuct, date)))

@retry(stop_max_attempt_number=3)
def download_da(date, procuct):
    import requests
    from fake_useragent import UserAgent
    ua = UserAgent(verify_ssl=False)
    header = {'User-Agent': ua.random,
              'Referer': 'http://www.dce.com.cn/publicweb/quotesdata/memberDealPosiQuotes.html',
              'Cookie': 'JSESSIONID=266ECD4AD2D3F048353269AB1EE7388B; WMONID=z0QY4IwsgT5'}
    d = {'memberDealPosiQuotes.variety': procuct,
         'memberDealPosiQuotes.trade_type':'0',
         'year': date[:4],
         'month': str(int(date[4:6]) - 1),
         'day': date[-2:],
         'contract.contract_id':'all',
         'contract.variety_id': procuct,
         'exportFlag':'excel'}
    print(d)
    r = requests.post('http://www.dce.com.cn/publicweb/quotesdata/exportMemberDealPosiQuotesData.html', headers=header, data=d)
    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), '{}_xlsdata/dafe{}.xls'.format(procuct, date)), 'wb') as fp:
        for chunk in r.iter_content(chunk_size=1024):
            fp.write(chunk)


def get_datfile(begindate, enddate, product, exchange):
    import urllib
    date_l = (datetime.strftime(x, '%Y%m%d') for x in pd.date_range(start=begindate, end=enddate))
    for date in date_l:
        print(date)
        if exchange == 'sh':
            dictionary = '{}_datdata'.format(product)
            if not os.path.exists(dictionary):
                os.makedirs(dictionary)
            try:
                download_sh(date, product)
            except urllib.error.HTTPError as e:
                print(e)
                continue
        if exchange == 'da':
            dictionary = '{}_xlsdata'.format(product)
            if not os.path.exists(dictionary):
                os.makedirs(dictionary)
            try:
                download_da(date, product)
            except urllib.error.HTTPError as e:
                print(e)
                continue

def gener_traindata(filepath):
    part_train_data = []
    for f in os.listdir(filepath):
        print(f)
        part_train_data.append(gen_rt_db(3, os.path.join(filepath, f)))
        break
    part_final_result = pd.DataFrame(part_train_data)
    t = part_final_result.pop('data_time')
    part_final_result.insert(0, 'data_time', t)
    part_final_result.sort_values(ascending=True, by='data_time')
    # print(part_final_result.dtypes)
    return part_final_result


def handle_senti():
    df = pd.read_csv('FutrueSentiment.csv')
    # print(df.dtypes)
    # df.sort_values(ascending=True, by='data_time')
    df.loc[:,'price_change_1'] = df.loc[:,'price'].rolling(2).apply(lambda x: x[1]-x[0])
    df.loc[:, 'price_change_3'] = df.loc[:, 'price'].rolling(4).apply(lambda x: x[3] - x[0])
    df.loc[:, 'price_change_5'] = df.loc[:, 'price'].rolling(6).apply(lambda x: x[5] - x[0])
    df.loc[:,'price_changerate_1'] = df.loc[:,'price'].rolling(2).apply(lambda x: (x[1]-x[0])/x[0])
    df.loc[:, 'price_changerate_3'] = df.loc[:, 'price'].rolling(4).apply(lambda x: (x[3] - x[0])/x[0])
    df.loc[:, 'price_changerate_5'] = df.loc[:, 'price'].rolling(6).apply(lambda x: (x[5] - x[0])/x[0])
    df.loc[:, 'price_changerate_1_label'] = df.loc[:, 'price_changerate_1'].shift(-1)
    df.loc[:, 'price_changerate_3_label'] = df.loc[:, 'price_changerate_3'].shift(-3)
    df.loc[:, 'price_changerate_5_label'] = df.loc[:, 'price_changerate_5'].shift(-5)
    df.loc[:,'label_1'] =  list(map(lambda x: 1 if x > 0 else (-1 if x < 0 else 0) , df.loc[:,'price_change_1']))
    df.loc[:, 'label_3'] = list(map(lambda x: 1 if x > 0 else (-1 if x < 0 else 0), df.loc[:, 'price_change_3']))
    df.loc[:, 'label_5'] = list(map(lambda x: 1 if x > 0 else (-1 if x < 0 else 0), df.loc[:, 'price_change_5']))
    df.loc[:, 'label_1'] = df.loc[:, 'label_1'].shift(-1)
    df.loc[:, 'label_3'] = df.loc[:, 'label_3'].shift(-3)
    df.loc[:, 'label_5'] = df.loc[:, 'label_5'].shift(-5)
    return df

def func2(x, li):
    return 'else' if x not in li else x

def id_onehot(df, interupt_date):
    df_part = df[df.data_time >= interupt_date]
    cols = [col for col in df_part.columns if '_id' in col]
    df_copy = df
    for col in cols:
        print(col)
        x = df_part.groupby(col)['data_time'].nunique().sort_values(ascending=False).head(14)
        y = list(x.index)
        df_copy[col] = df[col].apply(lambda x:func2(x, y))
        df_copy = pd.concat([df_copy, pd.get_dummies(df_copy[col], prefix='onehot_%s'%col[:-3])], axis=1)
    df_copy.drop(cols, axis=1, inplace=True)
    return df_copy





if __name__ == '__main__':
    # get_datfile('2018/04/01', '2018/05/21', 'j', 'da')
    # exit()
    # part_final_result = gener_traindata(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'rb_datdata'))
    part_final_result = gener_traindata(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'i_xlsdata'))
    sentiment_result = handle_senti()
    final_train = pd.merge(part_final_result, sentiment_result, on='data_time', how='outer').sort_values(ascending=True, by='data_time')
    # final_train = id_onehot(final_train,'2015-01-01')
    # final_train = pd.merge(pd.read_csv('FutrueSentiment.csv')[['data_time','DEA', 'EMA5']], final_train, on='data_time', how='outer')

    # final_train.to_csv('traindata_full.csv',index=False)
    # handle_for_lstm(5)
    # gen_rt_db(5, 'shfe20180410.dat')
