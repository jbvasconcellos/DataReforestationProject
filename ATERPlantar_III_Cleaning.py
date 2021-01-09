import pandas as pd
import numpy as np
from functools import reduce

#=============================================================================#

# Load data:
path = r'C:\Users\joaob\Dropbox\Trampos\Rioterra\Analises'
file = path + '\Original_ATERPlantar_III.xlsx'
sheets_list = ['CAD_AGRICULTOR', 'CAD_PROPRIEDADE', 'CAD_INFORMACOES', \
    'CAD_DADOS_RL', 'CAD_DADOS_APP', 'CAD_INF_COMPROVADAS', \
    'CAD_PLA_SELECAO']
base = pd.read_excel(file, sheet_name=sheets_list, encoding='latin1')
# display(base.keys())

# COLUMNS TRANSLATION DICTIONARY
translate = {'NOME DA PROPRIEDADE':'Farm', 'COORDENADA X UTM':'X_UTM',
'COORDENADA Y UTM':'Y_UTM', 'MUNICIPIO':'County',
'VISITAS':'Num_Visits', 'ATIVIDADE PRINCIPAL DA PROPRIEDADE':'Main_Activity',
'ATIVIDADE SECUNDÁRIA DA PROPRIEDADE':'Secondary_Activity', 'Estacas':'Piles',
'Mourões':'Columns', 'Catracas':'Tensioners',
'Bolas de Arame':'Wire_rolls', 'Situaçãdo do Isolamento':'Isolation',
'Quantidade de Calcário em Tonelada':'Carbonate',
'Total de área de APP comprovada':'APP_Area',
'Total de área de RL comprovada':'RL_Area',
'Total de Área Total Comprovada':'Total_Area',
'ISOLAMENTO CONCLUIDO':'Concluded_Isolation',
'ANO DE SELEÇÃO':'Year_Selection',
'ANO DE EXECUÇÃO':'Year_Implementation',
'SITUAÇÃO PROJ ANO':'Project_Status',
'ASSOCIAÇÃO / STTR':'Affiliation'}

# ===================== #
# PRODUCERS
df_Produtor= base['CAD_AGRICULTOR'].copy()
df_Produtor.drop(['APELIDO', 'RG', 'E-MAIL', 'TEL DE CONTATO 1', 'APK', 'TEL CONTATO 2'], axis=1, inplace=True)
df_Produtor.drop(0, inplace=True)
# display(df_Produtor.columns)

# FARMS
df_Propriedade = base['CAD_PROPRIEDADE'].copy()
df_Propriedade.drop(0, inplace=True)

df_Informacoes = base['CAD_INFORMACOES'].copy()
df_Informacoes.drop(['JÁ PARTICIPOU DE ALGUM PROJETO DO CESRIOTERRA?', 'EM QUAL PERÍODO?'], axis=1, inplace=True)
df_Informacoes.drop(0, inplace=True)
# Drop currently empty columns:
# df_Propriedade.columns

df_Properties = pd.merge(df_Propriedade, df_Informacoes, how='inner', on='COD_PRO')
df_Properties.drop(['COD_PLA', 'GLEBA', 'LINHA', 'KM', 'LOTE'], axis=1, inplace=True)
df_Properties.drop(1616, inplace=True)
df_Properties[['COD_PRO', 'COD_SEDE']] = df_Properties[['COD_PRO', 'COD_SEDE']].replace(r'\W', '', regex=True)
df_Properties.rename(translate, axis=1, inplace=True)
# df_Properties.columns

# ===================== #

# WORKS
df_VerifiedInfo = base['CAD_INF_COMPROVADAS'].copy()
# DROP CURRENTLY EMPTY COLUMNS
df_VerifiedInfo.drop(['Total de Área Realizada de Mecanização', 'Quantida de Hora Máquina Realizada'], axis=1, inplace=True)
df_VerifiedInfo.drop(0, inplace=True)
# DROP ROWS WITHOUT COD_PRO INDEX (these are empty rows)
# display(df_VerifiedInfo[df_VerifiedInfo['COD_PRO'].isna() == True].shape[0])
df_VerifiedInfo = df_VerifiedInfo[df_VerifiedInfo['COD_PRO'].isna() == False]
df_VerifiedInfo['COD_PRO'] = df_VerifiedInfo['COD_PRO'].str.replace(r'\W', '')
df_VerifiedInfo.reset_index(drop=True, inplace=True)

df_VerifiedInfo['COD_AGR'] = df_VerifiedInfo['COD_AGR'].str.replace(r'\W', '')
df_VerifiedInfo['COD_SEL'] = df_VerifiedInfo['COD_SEL'].str.replace(r'\W', '')

# No need for the following:
# DROP ROWS WITH COD_PRO VALUES NOT IN df['COD_PRO'] ANBD RESET INDEX
# df_VerifiedInfo = df_VerifiedInfo[df_VerifiedInfo['COD_PRO'].isin(df['COD_PRO'])]

df_RL = base['CAD_DADOS_RL'].copy()
df_APP = base['CAD_DADOS_APP'].copy()
df_RL = df_RL[['COD_AGR', 'COD_PRO', 'COD_SEL', 'COD_EXT', 'Necessita de Calcário?',
'Necessita isolamento?', 'Grupo', 'Cultura I', 'Cultura II',
'Quantidade de mudas', 'MODALIDADE', 'ANO DE EXECUCAO']]
df_APP = df_APP[['COD_AGR', 'COD_PRO', 'COD_SEL', 'COD_EXT', 'Necessita de Calcário?',
'Necessita isolamento?', 'Grupo', 'Cultura I', 'Cultura II',
'Quantidade de mudas', 'MODALIDADE', 'ANO DE EXECUCAO']]
df_RL.drop(0, inplace=True)
df_APP.drop(0, inplace=True)
# display(df_RL.head())
# display(df_APP.head())

df_RL.rename(columns={'Necessita de Calcário?':'Requires_Carbonate_RL',
'Necessita isolamento?':'Requires_Isolation_RL','Grupo':'Group_RL',
'Cultura I':'CultureI_RL', 'Cultura II':'CultureII_RL',
'Quantidade de mudas':'Seedlings_RL', 'MODALIDADE':'Method_RL',
'ANO DE EXECUCAO':'Year_RL'}, inplace=True)
df_APP.rename(columns={'Necessita de Calcário?':'Requires_Carbonate_APP',
'Necessita isolamento?':'Requires_Isolation_APP', 'Grupo':'Group_APP',
'Cultura I':'CultureI_APP', 'Cultura II':'CultureII_APP',
'Quantidade de mudas':'Seedlings_APP', 'MODALIDADE':'Method_APP',
'ANO DE EXECUCAO':'Year_APP'}, inplace=True)
# display(df_RL.head())
# display(df_APP.head())

# DROP ROWS WITHOUT COD_PRO INDEX (these are empty rows)
df_RL = df_RL[df_RL['COD_PRO'].isna() == False]
df_RL['COD_PRO'] = df_RL['COD_PRO'].str.replace(r'\W', '')
df_RL.reset_index(drop=True, inplace=True)
df_APP = df_APP[df_APP['COD_PRO'].isna() == False]
df_APP['COD_PRO'] = df_APP['COD_PRO'].str.replace(r'\W', '')
df_APP.reset_index(drop=True, inplace=True)
# display(df_RL.head())
# display(df_APP.head())

# df_RL = df_RL[df_RL['COD_PRO'].isin(df['COD_PRO'])]
# df_APP = df_APP[df_APP['COD_PRO'].isin(df['COD_PRO'])]

# REMOVE SPECIAL CHARACTERS FROM THE COD_SEL AND COD_PLA
df_RL['COD_AGR'] = df_RL['COD_AGR'].str.replace(r'\W', '')
df_RL['COD_SEL'] = df_RL['COD_SEL'].str.replace(r'\W', '')
df_APP['COD_AGR'] = df_APP['COD_AGR'].str.replace(r'\W', '')
df_APP['COD_SEL'] = df_APP['COD_SEL'].str.replace(r'\W', '')
# display(df_RL.head())
# display(df_APP.head())
# display(df_VerifiedInfo.head())
# display(df_RL.shape[0])
# display(df_APP.shape[0])

df_Selection = base['CAD_PLA_SELECAO'].copy()
df_Selection[['COD_AGR', 'COD_PRO', 'COD_SEL']] = df_Selection[['COD_AGR', 'COD_PRO', 'COD_SEL']].replace(r'\W', '', regex=True)


# DATAFRAMES:
# df_Properties: general information about each property
# display(df_Properties.columns)
# df_VerifiedInfo, df_RL and df_APP: informtaion about each reforested polygon

df_Areas = reduce(lambda left, right : \
pd.merge(left, right, how='inner', on=['COD_PRO', 'COD_SEL']), [df_VerifiedInfo, df_RL, df_APP, df_Selection])
# display(df_Areas.columns)
# display(df_Areas.shape[0])

drop = ['COD_AGR_x', 'COD_EXT_x', 'COD_AGR_y', 'COD_EXT_y', 'COD_AGR_x', 'COD_EXT_x',
     'COD_AGR_y', 'COD_EXT_y', 'COD EXT  EXECUÇÃO', 'ADESSÃO AO PROJETO',
       'TERMO DE RAD', 'TERMO DE PRA',  'MATERIAL DE CERCA ENTREGUE','CALCARIO ENTRGUE',
        'MECANIZACAO REALIZADA', 'MUDAS ENTREGUE', 'ATER/EDU/PRA/RAD e RAD+PRA', 'QNT VISITAS']
requires_material = ['Requires_Carbonate_RL', 'Requires_Isolation_RL',
'Requires_Carbonate_APP', 'Requires_Isolation_APP']

df_Areas.drop(drop, axis=1, inplace=True)
df_Areas.rename(columns=translate, inplace=True)

df_Areas[requires_material+['Concluded_Isolation']] = \
    df_Areas[requires_material+['Concluded_Isolation']].apply(lambda x: x.str.lower())
df_Areas[requires_material+['Concluded_Isolation']] = \
    df_Areas[requires_material+['Concluded_Isolation']].replace('-', np.nan)
# pd.unique(df_Areas[requires_material+['Concluded_Isolation']].values.ravel('K'))
material_categs = {'s':1, 'sim':1, 'n':0, 'não':0, '-':np.nan}
df_Areas[requires_material+['Concluded_Isolation']] =\
    df_Areas[requires_material+['Concluded_Isolation']].replace(material_categs).astype('category')
df_Areas[requires_material+['Concluded_Isolation']] = df_Areas[requires_material+['Concluded_Isolation']].fillna(0)
# df_Areas.info()

df_Areas['Isolation'].replace(['-', ';;'], np.nan, inplace=True)
df_Areas['Isolation'].unique()

# df_Areas[requires_material] = df_Areas[requires_material].replace(0, np.nan)

# STILL TO MANY MISSING VALUES!!
# 1089 rows with both RL and APP area missing out of 1681
# display(df_Areas[(df_Areas['APP_Area'].isna()) & (df_Areas['RL_Area'].isna()) & (df_Areas['Total_Area'].isna())].shape[0])
# display(df_Areas.shape[0])



# COLUMNS CATEGORIES
materials = ['Piles', 'Columns', 'Tensioners', 'Wire_rolls', 'Carbonate']
seedlings = ['Seedlings_APP','Seedlings_RL']
areas = ['APP_Area', 'RL_Area', 'Total_Area']

# REMOVE NULL AREAS
# ============================================================ #
# Exclude rows with all areas missing
# df_Areas[df_Areas[areas].isna().sum(axis=1) < 3].shape
df_Areas = df_Areas[df_Areas[areas].isna().sum(axis=1) < 3]

# Fill NaN areas
df_Areas[areas] = df_Areas[areas].fillna(0)

# Exclude rows with all null areas
df_Areas = df_Areas[df_Areas[areas].sum(axis=1) > 0]

# Replace null Total_Area with RL_area + APP_Area if the sum itself is not null
# PROBLEMATIC ROWS: 319 out of 758
# Rows s.t. the sum of areas does not correspond to the total area
# display(df_Areas[areas][df_Areas['Total_Area'] != df_Areas[['APP_Area', 'RL_Area']].sum(axis=1).round(2)].shape[0])
df_Areas['Sum_Areas'] = pd.Series(df_Areas[['APP_Area', 'RL_Area']].sum(axis=1))
# df_Areas.shape[0]
df_Areas['Total_Area'] = df_Areas[['Total_Area', 'Sum_Areas']].max(axis=1).round(2)
df_Areas.drop('Sum_Areas', axis=1, inplace=True)
# ============================================================ #

# NULL MATERIALS
# ============================================================ #
df_Areas[materials] = df_Areas[materials].fillna(0)

# Fix wrong labels for Requires_Isolation_RL and Requires_Carbonate_APP
df_Areas['Requires_Isolation'] = df_Areas[['Requires_Isolation_RL', 'Requires_Isolation_APP']].max(axis=1)
df_Areas['Requires_Carbonate'] = df_Areas[['Requires_Carbonate_RL', 'Requires_Carbonate_APP']].max(axis=1)

# Drop rows with null materials list based on Requires_Isolation APP and RL
# columns
# PROBLEMATIC ROWS
# Rows with nullfence materials values and yet either Requires_Isolation_RL or
# Requires_Isolation_APP equals to 1
# Rows 29, 266, 538, 604, 891, 1034, 1085, 1146, 1163, 1208
idx_NoIsolation = df_Areas[((df_Areas['Requires_Isolation_RL'] == 1) | \
    (df_Areas['Requires_Isolation_APP'] == 1)) & ((df_Areas[materials] == 0).all(axis=1))].index
# PROBLEMATIC ROWS
# Rows with null carbonate values and yet either Requires_Carbonate_RL or
# Requires_Carbonate_APP equals to 1
# 351,  377,  378,  519,  538,  556,  604,  633,  660,  891,  990, 1085, 1100,
# 1101, 1146, 1163, 1210
idx_NoCarb = df_Areas[((df_Areas['Requires_Carbonate_RL'] == 1) | \
    (df_Areas['Requires_Carbonate_APP'] == 1)) & (df_Areas['Carbonate'] == 0)].index



# PROBLEMATIC ROWS
# 528 areas that didn`t require any isolation, yet used some material
# df_Areas[(df_Areas['Requires_Isolation_RL'] == 0) & \
#     (df_Areas['Requires_Isolation_APP'] == 0) & (df_Areas[materials] != 0).any(axis=1)].shape[0]

# Fix wrong labels for Requires_Isolation_RL or Requires_Carbonate_APP = 1 but
# no material used
df_Areas.loc[idx_NoIsolation, 'Requires_Isolation'] = 0

# Analogous for carbonate
df_Areas.loc[idx_NoCarb, 'Requires_Carbonate'] = 0

# PROBLEMATIC ROW:
# row 853:
# df_Areas[df_Areas['Piles'] == '44 / 35'].index
# df_Areas[df_Areas['Columns'] == '6 / 3'].index
# Solving it in a very bad way:
df_Areas.loc[853, 'Piles'] = 44
df_Areas.loc[853, 'Columns'] = 6

df_Areas[['Piles', 'Columns', 'Tensioners', 'Wire_rolls']] = \
    df_Areas[['Piles', 'Columns', 'Tensioners', 'Wire_rolls']].astype(int)
df_Areas['Carbonate'] = df_Areas['Carbonate'].astype(float)
# df_Areas[materials].info()

df_Areas[['Piles', 'Columns', 'Tensioners', 'Wire_rolls']].sum(axis=1)

df_Areas.loc[df_Areas[['Piles', 'Columns', 'Tensioners', 'Wire_rolls']].sum(axis=1) > 0, 'Requires_Isolation'] = 1
# df_Areas['Requires_Isolation'][df_Areas[['Piles', 'Columns', 'Tensioners', 'Wire_rolls']].sum(axis=1) > 0].min()

# PROBLEMATIC AREAS VALUES: 42
# display(df_Areas[(df_Areas['APP_Area'] > 0) & (df_Areas['RL_Area'] > 0)].shape[0])
# THESE 42 VALUES ARE A REAL PROBLEM
# COMMENT:
# I'd like to fix wrong labels for Requires_Isolation_RL and
# Requires_Carbonate_APP in the following manner: if Requires_Isolation_RL and
# Requires_Isolation_APP are both null bit some material was used, then
# Requires_Material = 1. If RL_Area > 0 and APP_Area == 0, then
# Requires_Isolation_RL = 1. Analogously for APP areas. I can't implement the
# second step due to the dubious areas.

# EXPORT TO CSV FILE
df_Properties.to_csv('BaseATER_Properties.csv')
df_Areas.to_csv('BaseATER_Areas.csv')

# df_final['Isolation'].unique()
