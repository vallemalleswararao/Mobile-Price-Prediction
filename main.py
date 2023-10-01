import pandas as pd
import re
df=pd.read_csv("C:/Users/LENOVO/Documents/Python Projects/Mobile Price Prediction/mobile_price_data.csv")
df.head()
df.shape
df.dtypes
df.isna().sum()

df['mob_weight']=df['mob_weight'].str.replace('g','')

df['mob_height']=df['mob_height'].str.replace('mm','')
df['mob_depth']=df['mob_depth'].str.replace('mm','')
df['mob_width']=df['mob_width'].str.replace('mm','')
df['battery_power']=df['battery_power'].str.replace('mAh','')

df['mp_speed']=df['mp_speed'].str.replace('GHz','')
df['int_memory']=df['int_memory'].str.replace('GB','')
df['ram']=df['ram'].str.replace('GB','')
df['bluetooth'].value_counts()

df.drop('bluetooth',axis=1,inplace=True)
df.head()
df['num_cores'].value_counts()

df['num_cores']=df['num_cores'].map({'Octa Core':8,'Quad Core':4,'Single Core':1})
df['dual_sim'].value_counts()

df.drop('dual_sim',axis=1,inplace=True)

df['os'].unique()

df['os']=df['os'].str.replace(r'^Android\s[a-zA-Z]*\s?',r'',regex=True)
df['os'].unique()
df['os'].unique()
df[['os','hoax1','hoax2']]=df['os'].str.partition('.')
df.drop(['hoax1','hoax2'],axis=1,inplace=True)
df['os'].value_counts()
df['mobile_price']=df['mobile_price'].replace(['₹',','],'',regex=True)
df.head()
df['mobile_color'].unique()
df[['dummy1','dummy2','mobile_color']]=df['mobile_color'].str.rpartition(' ')
df.drop(['dummy1','dummy2'],axis=1,inplace=True)
df['mobile_color'].value_counts()
df['mobile_color']=df['mobile_color'].replace({'Greener':'Green','white':'White','gold':'Gold','Gray':'Grey'})
df['mobile_color'].value_counts()
df[['mobile_name','dummy1','dummy2']]=df['mobile_name'].str.partition(' ')
df.drop(['dummy1','dummy2'],axis=1,inplace=True)
df['mobile_name'].value_counts()
df['network'].head()
df['network']=df['network'].str.replace(' ','')
df['network']=df['network'].apply(lambda x: sorted(x.split(',')))
df['network'].value_counts()
from sklearn.preprocessing import MultiLabelBinarizer

mlb=MultiLabelBinarizer()
dg=pd.DataFrame(mlb.fit_transform(df['network']),columns=mlb.classes_,index=df.index)
dg
df=pd.merge(df,dg,left_index=True,right_index=True)
df.drop('network',axis=1,inplace=True)
df.head()
df['disp_size']=df['disp_size'].replace('\scm\s\(.+\)','',regex=True)
df['disp_size'].unique()
df['resolution'].unique()
df['resolution']=df['resolution'].replace('\s?[pP]ixel.*','',regex=True).replace('\s?[x*×]\s?','X',regex=True).replace('\$','',regex=True)
df['resolution'].unique()
df[['res_dim_1','sep','res_dim_2']]=df['resolution'].str.partition('X')
df.drop('sep',axis=1,inplace=True)
df['res_dim_1'].unique(),df['res_dim_2'].unique()
df['res_dim_1']=df['res_dim_1'].astype(int)
df['res_dim_2']=df['res_dim_2'].astype(int)
df['res_dim_1'].dtype,df['res_dim_2'].dtype
df['resolution']=df['res_dim_1']*df['res_dim_2']
df['resolution'].unique()
df.drop(['res_dim_1','res_dim_2'],axis=1,inplace=True)
df['p_cam']
df['p_cam_max']=[x[0:2].replace('M','') for x in df['p_cam']]
df['p_cam_count'] = [x.count('MP') for x in df['p_cam']]
df['f_cam']
df['f_cam_max']=[x[0:2].replace('M','') for x in df['f_cam']]
df['f_cam_count'] = [x.count('MP') for x in df['f_cam']]
df.drop(['f_cam','p_cam'],axis=1,inplace=True)
df.head()
df_mobile_name=pd.get_dummies(df['mobile_name'],dtype=int)
df_mobile_name
df_mobile_color=pd.get_dummies(df['mobile_color'],dtype=int)
df_mobile_color
df=pd.concat([df,df_mobile_name,df_mobile_color],axis=1)
df
df.drop(['mobile_name','mobile_color'],axis=1,inplace=True)
df.dtypes
df.select_dtypes(include='number').columns
df.select_dtypes(include='object').columns
int_col_list=['mobile_price', 'os',  'int_memory', 'ram',
       'battery_power',
       'p_cam_max', 'f_cam_max']
float_col_list=['disp_size','mp_speed', 'mob_height', 'mob_width', 'mob_depth', 'mob_weight']
df[int_col_list]=df[int_col_list].astype(int)
df[float_col_list]=df[float_col_list].astype(float)
df.dtypes.value_counts()
# Assuming your cleaned DataFrame is named df, and you want to save it to a file named "cleaned_mobile_data.csv"
df.to_csv("cleaned_mobile_data.csv", index=False)
