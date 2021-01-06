import pandas as pd
import numpy as np
from functools import reduce

#=============================================================================#

# Load data:
path = r'C:\Users\joaob\Dropbox\Trampos\Rioterra\Analises'
file = path + '\Original_ATERPlantar_III.xlsx'
base = pd.read_excel(file, sheet_name=None, encoding='latin1')
# display(base.keys())

# ===================== #
# PRODUCERS
df_Produtor= base['CAD_AGRICULTOR'].copy()
# display(df_Produtor.columns)
df_Produtor.drop(['APELIDO', 'RG', 'E-MAIL', 'TEL DE CONTATO 1', 'APK', 'TEL CONTATO 2'], axis=1, inplace=True)
df_Produtor.drop(0, inplace=True)

# ===================== #
# FARMS
df_Propriedade = base['CAD_PROPRIEDADE'].copy()
df_Propriedade.drop(0, inplace=True)

df_Informacoes = base['CAD_INFORMACOES'].copy()
df_Informacoes.drop(0, inplace=True)
# Drop currently empty columns:
df_Informacoes.drop(['JÁ PARTICIPOU DE ALGUM PROJETO DO CESRIOTERRA?', 'EM QUAL PERÍODO?'], axis=1, inplace=True)


df = pd.merge(df_Propriedade, df_Informacoes, how='inner', on='COD_PRO')
# Drop empty row:
df.drop(1616, inplace=True)


# ===================== #

# WORKS
df_VerifiedInfo = base['CAD_INF_COMPROVADAS'].copy()
df_VerifiedInfo.drop(0, inplace=True)
# DROP CURRENTLY EMPTY COLUMNS
df_VerifiedInfo.drop(['Total de Área Realizada de Mecanização', 'Quantida de Hora Máquina Realizada'], axis=1, inplace=True)
# DROP ROWS WITHOUT COD_PRO INDEX
df_VerifiedInfo = df_VerifiedInfo[df_VerifiedInfo['COD_PRO'].isna() == False ]
# DROP ROWS WITH COD_PRO VALUES NOT IN df['COD_PRO'] ANBD RESET INDEX
df_VerifiedInfo = df_VerifiedInfo[df_VerifiedInfo['COD_PRO'].isin(df['COD_PRO'])]
df_VerifiedInfo.reset_index(drop=True, inplace=True)

df_RL = base['CAD_DADOS_RL'].copy()
df_APP = base['CAD_DADOS_APP'].copy()
df_RL.drop(0, inplace=True)
df_APP.drop(0, inplace=True)
# df_APP.columns

df_RL = df_RL[['COD_AGR', 'COD_PRO', 'COD_SEL', 'COD_EXT', 'Necessita de Calcário?',
        'Necessita isolamento?', 'Grupo', 'Cultura I', 'Cultura II',
        'Quantidade de mudas', 'MODALIDADE', 'ANO DE EXECUCAO']]
df_RL.rename(columns={'Necessita de Calcário?':'Requires_Carbonate_RL',
        'Necessita isolamento?':'Requires_Isolation_RL','Grupo':'Group_RL',
        'Cultura I':'CultureI_RL', 'Cultura II':'CultureII_RL',
        'Quantidade de mudas':'Seedlings_RL', 'MODALIDADE':'Method_RL',
        'ANO DE EXECUCAO':'Year_RL'}, inplace=True)
# df_RL.columns

df_APP = df_APP[['COD_AGR', 'COD_PRO', 'COD_SEL', 'COD_EXT', 'Necessita de Calcário?',
        'Necessita isolamento?', 'Grupo', 'Cultura I', 'Cultura II',
        'Quantidade de mudas', 'MODALIDADE', 'ANO DE EXECUCAO']]
df_APP.rename(columns={'Necessita de Calcário?':'Requires_Carbonate_APP',
        'Necessita isolamento?':'Requires_Isolation_APP', 'Grupo':'Group_APP',
        'Cultura I':'CultureI_APP', 'Cultura II':'CultureII_APP',
        'Quantidade de mudas':'Seedlings_APP', 'MODALIDADE':'Method_APP',
        'ANO DE EXECUCAO':'Year_APP'}, inplace=True)
# df_APP.columns

df_RL = df_RL[df_RL['COD_PRO'].isna() == False]
df_RL = df_RL[df_RL['COD_PRO'].isin(df['COD_PRO'])]
df_RL.reset_index(drop=True, inplace=True)
df_APP = df_APP[df_APP['COD_PRO'].isna() == False]
df_APP = df_APP[df_APP['COD_PRO'].isin(df['COD_PRO'])]
df_APP.reset_index(drop=True, inplace=True)

df_final = reduce(lambda left, right : \
pd.merge(left, right, how='inner', on='COD_PRO'), [df, df_VerifiedInfo, df_RL, df_APP])

# base['CAD_PLA_SELECAO'].head()
# base['CAD_PLA_SELECAO'][['COD_PRO', 'ISOLAMENTO CONCLUIDO']]
df_final = pd.merge(df_final, base['CAD_PLA_SELECAO'][['COD_PRO', 'ISOLAMENTO CONCLUIDO']], how='inner', on='COD_PRO')
# df_final.columns
drop = ['COD_EXT_x', 'COD_MUN', 'COD_SEDE', 'CAD_PLA', 'COD_EXT_y', 'NOME DA ASSOCIÇÃO', 'COD_AGR_x', 'COD_SEL_x',
       'COD_EXT_x', 'COD_AGR_y', 'COD_SEL_y', 'COD_EXT_y', 'COD_AGR', 'COD_SEL', 'COD_EXT']
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
       'ISOLAMENTO CONCLUIDO':'Concluded_Isolation'}

df_final.drop(drop, axis=1, inplace=True)
df_final.rename(columns=translate, inplace=True)
df_final['Isolation'].replace(['-', ';;'], np.nan, inplace=True)
df_final['Concluded_Isolation'].replace({'S':1, 'N':0}, inplace=True)

required_material = ['Requires_Carbonate_RL', 'Requires_Isolation_RL',
    'Requires_Carbonate_APP', 'Requires_Isolation_APP']
df_final[required_material] = df_final[required_material].replace(0, np.nan)
df_final[required_material] = df_final[required_material].replace({'S':1, 'N':0, '-':np.nan}).astype('category')
#ATTENTION HERE!
# compiling the previous lines in the wrong order may eliminate 0
# values replacing N


# df_final.head()
df_final.info()
df_final.to_csv('Base_ATER.csv')

# df_final['Isolation'].unique()


# Index = pd.DataFrame([df_VerifiedInfo['COD_PRO'], df['COD_PRO']]).transpose()
# Index.columns = ['Ver', 'df']
#
#
# Index.tail(20)
#
# for key, row in Index.iterrows():
#     if (row['Ver'] != row['df']):
#         print(row)
#         break
