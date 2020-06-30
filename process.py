import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm
import gc

# 欠損値を０で補完後、ラベルエンコーディング
def label_encoding_zero(train,test,feature):

	print('Before fillna: ', train[feature].isna().sum())
	print('fllna for: ', feature)
	train[feature] = train[feature].fillna('0')
	print('After fillna', train[feature].isna().sum())

	print('Before fillna: ', test[feature].isna().sum())
	print('fllna for: ', feature)
	test[feature] = test[feature].fillna('0')
	print('After fillna', test[feature].isna().sum())


	data = train.append(test)[feature]

	label_encoder = LabelEncoder()
	label_encoder.fit(data.values)

	train[feature] = label_encoder.transform(train[feature])
	test[feature] = label_encoder.transform(test[feature])

# 欠損値を最頻値で補完後、ラベルエンコーディング
def label_encoding_mode(train,test,feature):

	print('Before fillna: ', train[feature].isna().sum())
	print('fllna for: ', feature)
	train[feature] = train[feature].fillna('0')
	print('After fillna', train[feature].isna().sum())

	print('Before fillna: ', test[feature].isna().sum())
	print('fllna for: ', feature)
	test[feature] = test[feature].fillna('0')
	print('After fillna', test[feature].isna().sum())


	data = train.append(test)[feature]

	label_encoder = LabelEncoder()
	label_encoder.fit(data.values)

	train[feature] = label_encoder.transform(train[feature])
	test[feature] = label_encoder.transform(test[feature])


def fill_na_zero(train,test,feature):
	train[feature] = train[feature].fillna(0)
	test[feature] = test[feature].fillna(0)

def fill_na_median(train,test,feature):
	train[feature] = train[feature].fillna(train[feature].median())
	test[feature] = test[feature].fillna(test[feature].median())


def fill_na_mean(train,test,feature):
	train[feature] = train[feature].fillna(train[feature].mean())
	test[feature] = test[feature].fillna(test[feature].mean())

def fill_na_mode(train,test,feature):
	train[feature] = train[feature].fillna(train[feature].mode()[0])
	test[feature] = test[feature].fillna(test[feature].mode()[0])

def fill_na_nothing(train,test,feature):
	train[feature] = train[feature].fillna('Nothing')
	test[feature] = test[feature].fillna('Nothing')

# 0を他の特徴量の値で補完
def fill_zero_hb(genba,feature_to,feature_from):
	indices = np.array(genba[genba[feature_to]==0].index)

	for i in indices:
		genba.loc[i,feature_to] = genba.loc[i,feature_from]


def set_land_price(df):
	df['tc_mseki'] = df['tc_mseki'].fillna(df['tc_mseki'].mean())
	df['tt_mseki'] = df['tt_mseki'].fillna(0)

	df['land_price'] = df['tc_mseki'] * (df['koji_hb']+df['kijun_hb']/2)


# 路線名をグループ分け
def set_rosen_nm1(df):
	df.loc[df['rosen_nm1'].isin(['東武日光線','秩父鉄道秩父本線']), 'rosen_nm1'] = 'a'
	df.loc[df['rosen_nm1'].isin(['JR常磐線(上野～取手)','西武豊島線']), 'rosen_nm1'] = 'b'
	df.loc[df['rosen_nm1'].isin(['JR高崎線','埼玉新都市交通伊奈線【ニューシャトル】']), 'rosen_nm1'] = 'c'
	df.loc[df['rosen_nm1'].isin(['西武新宿線','西武池袋線']), 'rosen_nm1'] = 'd'
	df.loc[df['rosen_nm1'].isin(['JR東北本線【宇都宮線】','西武狭山線']), 'rosen_nm1'] = 'e'
	df.loc[df['rosen_nm1'].isin(['日暮里・舎人ライナー','首都圏新都市鉄道つくばエクスプレス']), 'rosen_nm1'] = 'f'


# フロアをシンプルに置換
def set_levelplan(df):
	df.loc[df.tt_mseki == 0, 'levelplan'] = '土地売り'

	print(df['levelplan'].isna().sum())
	df['levelplan'] = df['levelplan'].fillna(df['levelplan'].mode()[0])
	print(df['levelplan'].isna().sum())

	df.loc[df['levelplan'].isin(['3F/4DK','3F/4LDK','3F/4LDK+S','3F/5LDK']), 'levelplan'] = '3F'
	df.loc[df['levelplan'].isin(['3F/2LDK','3F/2LDK+2S','3F/2LDK+S','3F/3DK','3F/3LDK','3F/3LDK+2S','3F/3LDK+S']), 'levelplan'] = '3F'
	df.loc[df['levelplan'].isin(['2F/5DK','2F/5LDK']), 'levelplan'] = '2F'
	df.loc[df['levelplan'].isin(['2F/4DK','2F/4LDK','2F/4LDK+S']), 'levelplan'] = '2F'
	df.loc[df['levelplan'].isin(['2F/3LDK','2F/3LDK+2S','2F/3LDK+S']), 'levelplan'] = '2F'
	df.loc[df['levelplan'].isin(['2F/2LDK','2F/2LDK+S']), 'levelplan'] = '2F'
	df.loc[df['levelplan'].isin(['1F/4LDK','1F/4LDK+S','1F/5LDK']), 'levelplan'] = '1F'
	df.loc[df['levelplan'].isin(['1F/2LDK','1F/3LDK']), 'levelplan'] = '1F'
	df.loc[df['levelplan'].isin(['土地売り']), 'levelplan'] = '0F'


def set_kobetsu(df):
	total_kobetsu = ['kobetsu1','kobetsu2','kobetsu3', 'kobetsu4']

	df['ex_kobetsu1'] = df['kobetsu1']

	df['new_kobetsu'] = 0

	dct = {'Nothing':0,'高圧線下':-3,'信号前':-1,'信号近い':-1,'横断歩道前':-1,'踏切付近':-3,'ごみ置き場前':-2,
		'心理的瑕疵あり':-3,'計画道路':-2,'地役権有':1,'敷延2ｍ絞りあり':-2,'宅内高低差あり':-2,
		'嫌悪施設隣接':-3,'アパート南隣':-1,'街道沿い':-1,'交通量多い':-1,'裏道':-1,'行き止まり途中':-1,
		'行き止まり':-2,'車進入困難':-2,'前面道が坂途中':-2,'眺望良':3,'床暖房付':1,'エネファーム付':1,
		'角地':3,'二方路':2,'三方路':3}

	for kobetsu in total_kobetsu:

		for key, value in dct.items():
			df[kobetsu] = df[kobetsu].str.replace(key,value)

		df['new_kobetsu'] += df[kobetsu]

	df.loc[df['new_kobetsu'] > 3, 'new_kobetsu'] = 4
	df.loc[df['new_kobetsu'] < -3, 'new_kobetsu'] = -4


def set_hokakisei(df):

	total_hokakisei = ['hokakisei1','hokakisei2','hokakisei3', 'hokakisei4']


	dct = {'土地区画整理法第７６条':1,'景観法':-1,'景観地区':-1,'文化財保護法（埋蔵文化財）':-1,
	'文化財保護法':-1,'農地法届出要':-1,'農地法':-1,'国土法':0,'埋蔵文化財':-3,
	'公有地拡大推進法':-2,'河川法':-1,'55条許可':-2,'43条許可':-1,
	'東日本大震災復興特別区域法':0,'東日本震災復興特':0,'自然公園法':0,'航空法':-3,
	'７６条申請':0,'風致地区':-1,'公拡法':-1,'高さ最高限度有':-1,'安行出羽地区地区計画':0,
	'区画整理法':0,'都市再生特別措置法':-2,'特定空港周辺特別措置法':-1,'下水道法':0,'外壁後退':0,
	'土壌汚染対策法':0,'建築協定':0,'小規模宅地開発':1,'都市公園法':0,'生産緑地法':0,
	'特定都市河川浸水':0,'43条第1項ただし書き許可':0}

	for hokakisei in total_hokakisei:
		# print(df.groupby(hokakisei).count())
		df[hokakisei] = df[hokakisei].fillna('0')

		for key, value in dct.items():
			df[hokakisei] = df[hokakisei].str.replace(key,value)

	df['hokakisei'] = df['hokakisei1'] + df['hokakisei2'] + df['hokakisei3'] + df['hokakisei4']


	df.loc[df['hokakisei'] > 3, 'hokakisei'] = 4
	df.loc[df['hokakisei'] < -3, 'hokakisei'] = -4


def set_jigata(df):

	kobetsu = 'jigata'

	dct = {'不整形地':-1,'整形地':0,'敷地延長':-1,'間口狭・奥行長':-2,'間口狭':-1,'奥行長':-1,}

	for key, value in dct.items():
		df[kobetsu] = df[kobetsu].str.replace(key,value)

def set_road_st(df):

	kobetsu = 'road_st'

	dct = {'未舗装':-1,'歩道あり':1,'問題なし':0,'歩道\+緑地帯あり':1,}

	for key, value in dct.items():
		df[kobetsu] = df[kobetsu].str.replace(key,value)



def set_setsudo_kj(df):

	kobetsu = 'setsudo_kj'

	dct = {'良い':1,'普通':0,'悪い':-1,}

	for key, value in dct.items():
		df[kobetsu] = df[kobetsu].str.replace(key,value)


def set_water(df):

	kobetsu = 'gas'

	dct = {'都市ガス':1,'個別プロパン':-1,'集中プロパン':0,}

	for key, value in dct.items():
		df[kobetsu] = df[kobetsu].str.replace(key,value)


	kobetsu = 'usui'

	dct = {'宅内処理':-1,'公共下水':1,'側溝':0,'水路':0,}

	for key, value in dct.items():
		df[kobetsu] = df[kobetsu].str.replace(key,value)

	kobetsu = 'gesui'

	dct = {'公共下水':1,'個別浄化槽':-1,'集中浄化槽':0,'汲取式':-1,}

	for key, value in dct.items():
		df[kobetsu] = df[kobetsu].str.replace(key,value)


	kobetsu = 'josui'

	dct = {'公営':1,'私営':0,'井戸':-1,}

	for key, value in dct.items():
		df[kobetsu] = df[kobetsu].str.replace(key,value)


	df['water'] = df['gas'] + df['usui'] + df['gesui'] + df['josui']


	df.loc[df['water'] > 3, 'water'] = 4
	df.loc[df['water'] < -3, 'water'] = -4


def set_setsudo_hi(df):

	hogaku =['南','東＋南','西＋南','南＋東','南＋西','南＋北','南＋南東','南＋南西','南＋北東','南＋北西','北＋南','南東＋南','南西＋南','北東＋南','北西＋南']

	df.loc[df.setsudo_hi.isin(hogaku), 'setsudo_hi'] = '2'
	df.loc[(df.setsudo_hi != '2') & (df.setsudo_hi.str.contains('南')), 'setsudo_hi'] = '1'
	df.loc[(df.setsudo_hi != '2') & (df.setsudo_hi != '1') , 'setsudo_hi'] = '0'


def set_kyori(df):

	df['walk_kyori'] = 1000
	df['bus_kyori'] = 500
	df['car_kyori'] = 0

	df.loc[df['bas_toho1']=='徒歩','walk_kyori'] = df['eki_kyori1']
	df.loc[df['bas_toho1']=='バス','bus_kyori'] = df['eki_kyori1'] + df['teiho1']
	df.loc[df['bas_toho1']=='車','car_kyori'] = df['eki_kyori1']



def encode_df(train,test,feature):

	data = train.append(test)[feature]

	label_encoder = LabelEncoder()
	label_encoder.fit(data.values)

	train[feature] = label_encoder.transform(train[feature])
	test[feature] = label_encoder.transform(test[feature])



def encode_genba(train_genba,test_genba,feature):

	data = train_genba.append(test_genba)[feature]

	label_encoder = LabelEncoder()
	label_encoder.fit(data.values)

	train_genba[feature] = label_encoder.transform(train_genba[feature])
	test_genba[feature] = label_encoder.transform(test_genba[feature])


def cut(feature,arr):
	train[feature] = pd.cut(train[feature].values, arr, include_lowest=True)
	test[feature] = pd.cut(test[feature].values, arr, include_lowest=True)

	train[feature] = train[feature].astype(str)
	test[feature] = test[feature].astype(str)



def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

# From Kaggle notebook
def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 

    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)




# Initialize genba and goto --------------------------------------------------------------------------

train_genba = pd.read_csv('./dataset/re_train_genba.csv',encoding='shift_jis')
train_goto = pd.read_csv('./dataset/train_goto.csv')

train_genba['groups'] = np.arange(len(train_genba))

train_goto['pj_count'] = train_goto.groupby('pj_no')['pj_no'].transform('count')

test_genba = pd.read_csv('./dataset/re_test_genba.csv',encoding='shift_jis')
test_goto = pd.read_csv('./dataset/test_goto.csv')

test_genba['groups'] = np.arange(len(test_genba))

test_goto['pj_count'] = test_goto.groupby('pj_no')['pj_no'].transform('count')


train_genba = train_genba[train_genba['rosen_nm1'].isin(test_genba['rosen_nm1'].unique())]


# jukyo ----------------------------------------------------------------------------------

new = train_genba['jukyo'].str.split('市') 
train_genba['jukyo'] = new.apply(lambda x:x[0])


new = test_genba['jukyo'].str.split('市') 
test_genba['jukyo'] = new.apply(lambda x:x[0])

encode_genba(train_genba,test_genba,'jukyo')


# # ido & keido ----------------------------------------------------------------------------------

# train_ido_keido = pd.read_csv('./dataset/train_ido_keido.csv')

# train_ido_keido = train_ido_keido.astype(float)

# train_genba['ido'] = train_ido_keido['ido'] - 35
# train_genba['keido'] = train_ido_keido['keido'] - 139


# train_genba['ido'] = pd.cut(train_genba['ido'].values, 160, include_lowest=True)
# train_genba['ido'] = train_genba['ido'].astype(str)


# train_genba['keido'] = pd.cut(train_genba['keido'].values, 160, include_lowest=True)
# train_genba['keido'] = train_genba['keido'].astype(str)


# test_genba['ido'] = str(0.5)
# test_genba['keido'] = str(0.5)


# encode_genba(train_genba,test_genba,'ido')
# encode_genba(train_genba,test_genba,'keido')


# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
# 	print(train_genba.groupby('ido')['ido'].count())
# 	print(train_genba.groupby('keido')['keido'].count())

# import sys
# sys.exit('Finished')


# cut('fukuin',[0,3.99,4.99,5.99,6.99,7.99,11.99,15.99,26])



# eki_nm1 -----------------------------------------------------------------------------

eki_table = pd.read_csv('./dataset/eki_table.csv',encoding='shift_jis')

for old_eki_nm1, new_eki_nm1 in tqdm(zip(eki_table.old_eki_nm1.values, eki_table.new_eki_nm1.values)):
	train_genba.loc[train_genba['eki_nm1'] == old_eki_nm1, 'eki_nm1'] = new_eki_nm1

for old_eki_nm1, new_eki_nm1 in tqdm(zip(eki_table.old_eki_nm1.values, eki_table.new_eki_nm1.values)):
	test_genba.loc[test_genba['eki_nm1'] == old_eki_nm1, 'eki_nm1'] = new_eki_nm1

# train_genba['old_eki_nm1'] = train_genba['eki_nm1']

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
# 	print(train_genba['eki_nm1'])

# train_genba.to_csv('re_re_train_genba.csv',index=False)
# test_genba.to_csv('re_re_train_genba.csv',index=False)


# kyori -------------------------------------------------------------------------------------

fill_na_median(train_genba,test_genba,'gk_yoc_tm')
fill_na_median(train_genba,test_genba,'gk_chu_tm')
fill_na_median(train_genba,test_genba,'gk_sho_tm')


train_genba['kyori'] = train_genba['gk_yoc_tm'] + train_genba['gk_chu_tm'] + train_genba['gk_sho_tm']
test_genba['kyori'] = test_genba['gk_yoc_tm'] + test_genba['gk_chu_tm'] + test_genba['gk_sho_tm']


# Merge -------------------------------------------------------------------------------------

train = pd.merge(train_genba, train_goto, how='inner', on='pj_no')
test = pd.merge(test_genba, test_goto, how='inner', on='pj_no')

# Set features -------------------------------------------------------------------------------

numbers_mean = ['rosenka_hb','minmenseki','gk_sho_kyori','gk_chu_kyori', \
				'tt_mseki_max_hb','tc_mseki_avg_hb','gk_yoc_tm','gk_sho_tm','gk_chu_tm','fi4m_kyori', \
				'fi3m_kyori','tc_mseki_max_hb','mseki_yt_hb','mseki_rd_hb','mseki_dp_hb','road1_fi', \
				'road1_mg','road2_fi','road2_mg','tt_mseki_avg_hb','kempei1','yoseki1', \
				'chiseki_js_hb','chiseki_kb_hb','road3_fi','road3_mg','road4_fi','road4_mg', \
				'tc_mseki_min_hb','tt_mseki_min_hb'
				]

numbers_median = ['magutchi','teiho1','teiho2','kyori']

numbers_zero = ['bus_hon','tateuri_su', 'tochiuri_su','joken_su','hy1f_date_su', 'eki_kyori1', \
				'hy2f_date_su', 'hy3f_date_su','kaoku_hb','yheki_kotei','niwasaki']

categories_mode = ['bas_toho1','kaoku_um', \
					'road1_sb','yheki_umu', \
					'kborjs','hw_status', 'toshikuiki1','toshikuiki2','kodochiku','chikukeikaku', \
					'keikakuroad','kaihatsukyoka','t53kyoka','hokakyoka','bokachiiki', \
					'yoto1','yoto2','hiatari','bus_yohi','jukyo', 'yheki_yohi', \
					'road1_hk','road2_hk','road3_hk', 'road4_hk']

# categories_mode = ['bas_toho1','jigata']

categories_zero = ['hokakisei1','hokakisei2','hokakisei3','hokakisei4','kinshijiko','fi4m_yohi', \
					'road2_sb','road3_sb','road4_sb', \
					'fi3m_yohi','shu_jutaku','shu_bochi','shu_sogi' ,'shu_kokyo','shu_kaido',\
					'rs_e_kdate3','rs_e_parking','rs_e_m_ari','rs_e_tahata', 'rs_e_zoki', \
					'rs_w_kdate3','rs_w_parking','rs_w_m_ari','rs_w_tahata', 'rs_w_zoki', \
					'rs_s_kdate3','rs_s_parking','rs_s_m_ari','rs_s_tahata', 'rs_s_zoki', \
					'rs_n_kdate3','rs_n_parking','rs_n_m_ari','rs_n_tahata', 'rs_n_zoki',\
					'sho_conv','sho_super','sho_shoten','sho_market', \
					'shu_park','shu_shop','shu_factory','shu_tower','shu_line_ari','shu_line_nashi', \
					'shu_soon','shu_zoki','shu_highway','bastei_nm1','shu_hvline']


others = ['kobetsu1','kobetsu2','kobetsu3','kobetsu4','setsudo_hi', 'jigata', 'rosen_nm1', \
			'setsudo_kj','road_st','gas','usui','gesui','josui','fukuin','garage','eki_nm1','pj_count']


base = ['tc_mseki','tt_mseki','koji_hb','kijun_hb', 'levelplan', 'groups']

lst = base + categories_zero + categories_mode + numbers_mean + numbers_zero + numbers_median + others


test = test[lst]

lst.append('keiyaku_pr')
train = train[lst]

# # # jukyo to reorder -----------------------------------------------------------------------

# print(train['jukyo'])

# df_for_sorting = train.groupby('jukyo',as_index=False).mean()

# # print(df_for_sorting['jukyo'])

# df_for_sorting = pd.DataFrame(data={'jukyo':df_for_sorting['jukyo'],'mean_keiyaku_pr':df_for_sorting['keiyaku_pr']}, \
# 							index=range(df_for_sorting.shape[0]))

# df_for_setting = df_for_sorting.sort_values(by='mean_keiyaku_pr',ascending=True)
# df_for_setting.reset_index(drop=True, inplace=True)
# # print(df_for_setting['hu'])

# df_for_setting['new_jukyo'] = pd.Series(range(df_for_setting.shape[0]))

# # print(df_for_setting)

# train['new_jukyo'] = train['jukyo']

# # print(train[['jukyo','new_jukyo']])

# # Assign new_feature to each feature in TRAIN
# for feature_val,new_feature_val in zip(df_for_setting['jukyo'].values,df_for_setting['new_jukyo'].values):
# 	train.loc[train['jukyo'] == feature_val, 'new_jukyo'] = new_feature_val

# # print(train[['jukyo','new_jukyo']])

# train['jukyo'] = train['new_jukyo']
# train.drop('new_jukyo',axis=1,inplace=True)


# test['new_jukyo'] = test['jukyo']

# # Assign new_feature to each feature in TEST
# for feature_val,new_feature_val in zip(df_for_setting['jukyo'].values,df_for_setting['new_jukyo'].values):
# 	test.loc[test['jukyo'] == feature_val, 'new_jukyo'] = new_feature_val

# test['jukyo'] = test['new_jukyo']
# test.drop('new_jukyo',axis=1,inplace=True)

# print(train['jukyo'])


# Count encoding -------------------------------------------------------------------------------------
combined_df = pd.concat([train,test],ignore_index=True,sort=False)

features = ['koji_hb','kijun_hb','jukyo','eki_nm1','rosen_nm1','gas','gesui','usui','hiatari']

for feature in features:
	num_cat = combined_df.groupby(feature)[feature].transform('count')

	combined_df[str(feature) + '_count_encoded'] = num_cat


train = combined_df[:train.last_valid_index()+1]
test = combined_df[train.last_valid_index()+1:]

del combined_df; gc.collect()



# land_price -----------------------------------------------------------------------------

fill_na_zero(train,test,'tc_mseki')
fill_na_median(train,test,'koji_hb')

set_land_price(train)
set_land_price(test)



# rosen_nm1 ----------------------------------------------------------------------------------
fill_na_mode(train,test,'rosen_nm1')

set_rosen_nm1(train)
set_rosen_nm1(test)

encode_df(train,test,'rosen_nm1')


# levelplan -------------------------------------------------------------
fill_na_mode(train,test,'levelplan')

set_levelplan(train)
set_levelplan(test)

encode_df(train,test,'levelplan')
# ------------------------------------------------------------------------------------------------------

# df[['eki_kyori1','teiho1']] =  df[['eki_kyori1','teiho1']].astype(float)


# setsudo_hi ----------------------------------------------------------------------------------------------

set_setsudo_hi(train)
set_setsudo_hi(test)

fill_na_mode(train,test,'setsudo_hi')

encode_df(train,test,'setsudo_hi')


# kobetsu ---------------------------------------------------------------------------------------------

features = ['kobetsu1','kobetsu2','kobetsu3','kobetsu4']

for i in features:
	fill_na_nothing(train,test,i)


set_kobetsu(train)
set_kobetsu(test)

encode_df(train,test,'new_kobetsu')
encode_df(train,test,'ex_kobetsu1')


# hokakisei ----------------------------------------------------------------------------------------------

features = ['hokakisei1','hokakisei2','hokakisei3','hokakisei4']

for i in features:
	fill_na_mode(train,test,i)


set_hokakisei(train)
set_hokakisei(test)

encode_df(train,test,'hokakisei')


# jigata -----------------------------------------------------------------------
fill_na_mode(train,test,'jigata')

set_jigata(train)
set_jigata(test)

encode_df(train,test,'jigata')


# road_st -----------------------------------------------------------------------
fill_na_mode(train,test,'road_st')

set_road_st(train)
set_road_st(test)

encode_df(train,test,'road_st')


# setsudo_kj -----------------------------------------------------------------------
fill_na_mode(train,test,'setsudo_kj')

set_setsudo_kj(train)
set_setsudo_kj(test)

encode_df(train,test,'setsudo_kj')


# water ----------------------------------------------------------------------------------------------

features = ['josui','gesui','gas','usui']

for i in features:
	fill_na_mode(train,test,i)


set_water(train)
set_water(test)

encode_df(train,test,'water')


# fukuin ----------------------------------------------------------------------------------------------

fill_na_zero(train,test,'fukuin')

train['origin_fukuin'] = train['fukuin']
test['origin_fukuin'] = test['fukuin']

cut('fukuin',[0,3.99,4.99,5.99,6.99,7.99,11.99,15.99,26])

encode_df(train,test,'fukuin')


# garage ----------------------------------------------------------------------------------------------

fill_na_zero(train,test,'garage')

# print(train.groupby('garage').count())

cut('garage',[0,0.99,1.99,2.99,4.99,10.0])

encode_df(train,test,'garage')


# niwasaki ----------------------------------------------------------------------------------------------

fill_na_zero(train,test,'niwasaki')

train['niwasaki'] = train['niwasaki']
test['niwasaki'] = test['niwasaki']

train.drop(train[train['niwasaki'] > 900].index, inplace=True)

train.reset_index(drop=True, inplace=True)

cut('niwasaki',[0,0.99,1.99,2.99,3.99,4.99,5.99,6.99,7.99,8.99,9.99,10.99,11.99,14.99,19.99,33.01])


encode_df(train,test,'niwasaki')


train['target_encoded'], test['target_encoded'] = target_encode(train["jukyo"], 
                         test["jukyo"], 
                         target=train.keiyaku_pr, 
                         min_samples_leaf=3,
                         smoothing=10,
                         noise_level=0.01)



# eki_kyori ----------------------------------------------------------------------------------------------

fill_na_mode(train,test,'bas_toho1')
fill_na_zero(train,test,'teiho1')
fill_na_mean(train,test,'eki_kyori1')

set_kyori(train)
set_kyori(test)


# cut('walk_kyori',[0,1.01,3.01,5.01,8.01,10.01,15.01,20.01,25.01,30.01,50.01,70.01,90.01,1001])
# cut('bus_kyori',[0,5.01,10.01,15.01,20.01,25.01,30.01,40.01,51.01,501])
# cut('car_kyori',[0,0.99,5.01,8.01,10.01,15.01,21.01])


# encode_df(train,test,'walk_kyori')
# encode_df(train,test,'bus_kyori')
# encode_df(train,test,'car_kyori')



# re_koji ----------------------------------------------------------------------------------------------

# print('bas_toho1 for train: ', train[train['bas_toho1']=='車'].shape[0])
# print('bas_toho1 for test: ' , test[test['bas_toho1']=='車'].shape[0])

# fill_na_mean(train,test,'koji_hb')

# train['re_koji_hb'] = (train['koji_hb'] / 100).astype(int)
# test['re_koji_hb'] = (test['koji_hb'] / 100).astype(int)

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
# 	print(train.groupby('re_koji_hb')['re_koji_hb'].count())

# # cut('koji_hb',[111.99,199.99,299.99,399.99,499.99,599.99,699.99,799.99,899.99,999.99, \
# # 				1099.99,1199.99,1299.99,1399.99,1499.99,1599.99,1699.99,1799.99,1899.99, \
# # 				1999.99,2099.99,2199.99,1799.99,1799.99,1799.99,1799.99,1799.99,1799.9p, \


# cut('re_koji_hb',[199.99, 299.99, 399.99, 499.99, 599.99, 699.99, 799.99, 899.99, 999.99, 1099.99, 1199.99, \
# 				1299.99, 1399.99, 1499.99, 1599.99, 1699.99, 1799.99, 1899.99, 1999.99, 2099.99, 2199.99, 2299.99, \
# 				2399.99, 2499.99, 2599.99, 2699.99, 2799.99, 2899.99, 2999.99, 3099.99, 3199.99, 3299.99, 3399.99, \
# 				3499.99, 3599.99, 3699.99, 3799.99, 3899.99, 3999.99, 4099.99, 4199.99, 4299.99, 4399.99, 4499.99, \
# 				4599.99, 4699.99, 4799.99, 4899.99])


# encode_df(train,test,'re_koji_hb')


# total = 199.99

# lst = list()

# for i in range(50):
# 	lst.append(total)
# 	total+=100

# print(lst)

# lst.append()


# encode_df(train,test,'walk_kyori')



# All others-----------------------------------------------------------------------

for feature in numbers_zero:
	fill_na_zero(train,test,feature)

for feature in numbers_mean:
	fill_na_mean(train,test,feature)

for feature in numbers_median:
	fill_na_median(train,test,feature)

for feature in categories_zero:
	label_encoding_zero(train,test,feature)

for feature in categories_mode:
	label_encoding_mode(train,test,feature)


# rosenka_hb / 0.8 / 0.95 * tc_mseki ----------------------------------------------------------------------------------------------

train['new_rosenka'] = train['rosenka_hb'] / 0.8 / 0.95 * train['tc_mseki']
test['new_rosenka'] = test['rosenka_hb'] / 0.8 / 0.95 * test['tc_mseki']


# avg ----------------------------------------------------------------------------------------------

train['avg'] = (train['koji_hb'] + train['kijun_hb'] + (train['rosenka_hb']/0.8/0.95)) / 3
test['avg'] = (test['koji_hb'] + test['kijun_hb'] + (test['rosenka_hb']/0.8/0.95)) / 3


# re_level ----------------------------------------------------------------------------------------------

train['re_level'] = train['levelplan'] * np.sqrt(train['tt_mseki'])
test['re_level'] = test['levelplan'] * np.sqrt(test['tt_mseki'])


# aggreagate ----------------------------------------------------------------------------------------------

train['power'] = train['koji_hb'] * train['tt_mseki']
test['power'] = test['koji_hb'] * test['tt_mseki']

train['power2'] = train['koji_hb'] * train['re_level']
test['power2'] = test['koji_hb'] * test['re_level']

train['power4'] = train['tc_mseki'] * train['yheki_kotei'] * train['koji_hb']
test['power4'] = test['tc_mseki'] * test['yheki_kotei'] * test['koji_hb']

train['power5'] = train['koji_hb'] * train['origin_fukuin']
test['power5'] = test['koji_hb'] * test['origin_fukuin']

train['power6'] = train['koji_hb'] * train['niwasaki']
test['power6'] = test['koji_hb'] * test['niwasaki']


# 外れ値除去　-------------------------------------------------------------------------------------------------------

indices = train[(train['keiyaku_pr'] < 10000000) & (train['tt_mseki'] != 0)].index

train.loc[indices,'keiyaku_pr'] *= 10


# tt_mseki VS land_price
# 左上と左下一杯
train.drop(train[(train['land_price'] > 65000000) & (train['tt_mseki'] == 0)].index, inplace=True)
train.drop(train[(train['land_price'] < 8000000) & (train['tt_mseki'] == 0)].index, inplace=True)

# 上枠一杯
train.drop(train[(train['land_price'] > 75000000)].index, inplace=True)

# 中間２つ
train.drop(train[(train['tt_mseki'] > 40) & (train['tt_mseki'] < 63)].index, inplace=True)

# 右端2つ
#一つ目
train.drop(train[(train['tt_mseki'] > 140)].index, inplace=True)

#二つ目
train.drop(train[(train['land_price'] < 10000000) & (train['tt_mseki'] > 130) & (train['tt_mseki'] < 140)].index, inplace=True)


# x=(koji_hb+kijun_hb)/2, y=land_price
train.drop(train[((train['koji_hb']+train['kijun_hb'])/2 > 600000) & ((train['koji_hb']+train['kijun_hb'])/2 < 700000)].index, inplace=True)


# # # -------------------------------------------------------------------------------------------------------
# # # magutchi
train.drop(train[(train['magutchi'] < 2.0)].index, inplace=True)



# # # -------------------------------------------------------------------------------------------------------
# # # jukyo
train.drop(train[~train['jukyo'].isin(test['jukyo'].unique())].index, inplace=True)



train.reset_index(drop=True, inplace=True)


train_y = pd.DataFrame(columns=['keiyaku_pr'])
train_y['keiyaku_pr'] = train['keiyaku_pr']
train_y.to_csv('./dataset/train_y.csv',index=False,encoding='shift_jis')

print(train.shape)

train.to_csv('./dataset/filled_train_all.csv',index=False,encoding='shift_jis')

test.to_csv('./dataset/filled_test_all.csv',index=False,encoding='shift_jis')
