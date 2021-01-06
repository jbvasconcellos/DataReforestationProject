# %% codecell
# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from functools import reduce

%matplotlib inline
# %% codecell
# IMPORT DATA
df = pd.read_csv('Base_ATER.csv', index_col=0)

# SELECT AREAS WITH COMPLETED ISOLATION WORKS ONLY
df['Concluded_Isolation'] = df.loc[:, 'Concluded_Isolation'].astype('category')
df_Iso = df[(df['Isolation'] == 'CONCLUIDO') | (df['Concluded_Isolation'] == 1)].reset_index(drop=True)

# DROP A FEW COLUMNS WE WON'T USE
df_Iso.drop(['COD_PLA', 'Farm', 'Num_Visits', 'GLEBA', 'LINHA', 'KM', 'LOTE'], axis=1, inplace=True)

# FILL MISSING DATA
materials = ['Piles', 'Columns', 'Tensioners', 'Wire_rolls', 'Carbonate']
seedlings = ['Seedlings_APP','Seedlings_RL']
areas = ['APP_Area', 'RL_Area', 'Total_Area']

df_Iso[materials] = df_Iso[materials].fillna(0).astype(float)
df_Iso[areas] = df_Iso[areas].fillna(0).astype(float)
df_Iso = df_Iso[df_Iso['Total_Area'] > 0] #Removes problematic rows with area = 0
df_Iso.reset_index(drop=True, inplace=True)


# TRANSLATIONS
# According to Rioterra's data management coord. Felipe Uchoa,
# Olericulture = Horticulture and
# Fruticultura* = cocoa growing
activities = {
        'BOVINOCULTURA DE LEITE':'Dairy cattle',
        'BOVINOCULTURA DE CORTE':'Beef cattle',
        'BOVINOCULTURA MISTA':'Mixed cattle',
        'CAFEICULTURA':'Coffee cultivation',
        'LAVOURA':'Agriculture',
        'PISCICULTURA':'Fish farming',
        'FRUTICULTURA':'Fruit growing', 'FRUTICULTURA*':'Fruit growing',
        'HORTICULTURA':'Horticulture',
        'AGROINDUSTRIA':'Agroindustry',
        'ARRENDAMENTO':'Lease',
        'GRANJA AVICOLA':'Poultry',
        'GRANJA SUINA':'Pork',
        'SABOARIA':'Soap production',
        'OLERICULTURA':'Horticulture'
        }

df_Iso[['Main_Activity', 'Secondary_Activity']] = df_Iso[['Main_Activity', 'Secondary_Activity']].replace(activities)
df_Iso['Secondary_Activity'].fillna('None', inplace=True)

df_Iso.info()
# %% markdown
# ## Linear correlations
# %% codecell
display(df_Iso.corr().loc[:, materials])
# %% markdown
# INTERESTING RESULTS:
#
# - correlations between different materials, as expected
#
# - Largest correlation with areas: (both quite low)
#  - APP_Area and Piles (0.40)
#  - APP_Area and Wire_rolls (0.39)
#
#
# Why isn't the same tue for RL? Is the land type (RL, APP) determined by
# features not displayed above (topography, soil type, p.H. etc.) due to
# their exploitation rules?
# Significant correlation between seedlings and Piles, Wire_rolls
# Questions: how is a fence built? how is the amount of seedlings estimated?
# %% codecell
# FOCUS ON AREAS AND METERIALS, SEEDLINGS
display(df_Iso.corr().loc[areas, seedlings + materials])
# %% codecell
# MATERIALS PER TYPE OF LAND USE (RL AND APP)
# Very inefficient way of doing so many plots
fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(10,10))
fig.subplots_adjust(wspace=0.5, hspace=0.3, left=0.125, right=0.8, top=1.5, bottom=0.1)
fig.suptitle('Materials per area (x label)', y=1.5)

ax[0,0].scatter(y='Columns', x='APP_Area', data=df_Iso, color='r')
ax[0,0].set_title('Columns vs. APP area')
ax[0,0].set_ylabel('Num. of columns')
ax[0,0].set_xlabel('APP area')

ax[0,1].scatter(y='Columns', x='RL_Area', data=df_Iso, color='b')
ax[0,1].set_title('Columns vs. RL area')
ax[0,1].set_ylabel('Num. of columns')
ax[0,1].set_xlabel('RL area')

ax[1,0].scatter(y='Piles', x='APP_Area', data=df_Iso, color='r')
ax[1,0].set_title('Piles vs APP area')
ax[1,0].set_ylabel('Num. of piles')
ax[1,0].set_xlabel('APP area')

ax[1,1].scatter(y='Piles', x='RL_Area', data=df_Iso, color='b')
ax[1,1].set_title('Piles vs. RL area')
ax[1,1].set_ylabel('Num. of piles')
ax[1,1].set_xlabel('RL area')

ax[2,0].scatter(y='Wire_rolls', x='APP_Area', data=df_Iso, color='r')
ax[2,0].set_title('Wire_rolls vs APP Area')
ax[2,0].set_ylabel('Num. of wire rows')
ax[2,0].set_xlabel('APP area')

ax[2,1].scatter(y='Wire_rolls', x='RL_Area', data=df_Iso, color='b')
ax[2,1].set_title('Wire_rolls vs RL Area')
ax[2,1].set_ylabel('Num. of wire rolls')
ax[2,1].set_xlabel('RL area')

ax[3,0].scatter(y='Tensioners', x='APP_Area', data=df_Iso, color='r')
ax[3,0].set_title('Tensioners vs APP Area')
ax[3,0].set_ylabel('Num. of tensioners')
ax[3,0].set_xlabel('APP area')


ax[3,1].scatter(y='Tensioners', x='RL_Area', data=df_Iso, color='b')
ax[3,1].set_title('Tensioners vs RL Area')
ax[3,1].set_ylabel('Num. of tensioners')
ax[3,1].set_xlabel('RL area')

ax[4,0].scatter(y='Carbonate', x='APP_Area', data=df_Iso, color='r')
ax[4,0].set_title('Carbonate vs APP Area')
ax[4,0].set_ylabel('Kgs of carbonate')
ax[4,0].set_xlabel('APP area')

ax[4,1].scatter(y='Carbonate', x='RL_Area', data=df_Iso, color='b')
ax[4,1].set_title('Carbonate vs RL Area')
ax[4,1].set_ylabel('Kgs of carbonate')
ax[4,1].set_xlabel('RL area')

plt.show()
# %% markdown
# There is a peak of materials used in null areas. This may be bacause the value of each material in the dataframe might refer to the total amount used in each farm. That is, a farm may have both a RL and a separate APP area, and the values fior carbonate and fence materials may refer to the total amount of each material employed in the farm.
#
# This is probably affecting the correlation analysis and thus we should separate such cases.
# %% codecell
display('APP areas', df_Iso[df_Iso['APP_Area'] > 0].corr().loc['APP_Area', materials+seedlings])
display('RL areas', df_Iso[df_Iso['RL_Area'] > 0].corr().loc['RL_Area', materials+seedlings])
# %% markdown
# Separating the materials in the areas increase significantly the correlations as we exclude the left peak of points in the plots.
#
# We thus conclude that any analysis of Total_Area is of little or no meaning, and RL and APP areas ought to be considered separately.
# %% markdown
# ## Polynomial correlations
# %% codecell
df_IsoLog = df_Iso.apply(lambda x : np.log(x + 0.00001) if x.name in areas else x)
display(df_IsoLog.corr().loc[:, materials])
# %% markdown
# Added 0.00001 because I got a divided by zero error message:
# ```
# 679: RuntimeWarning: divide by zero encountered in log
# result = getattr(ufunc, method)(*inputs, **kwargs)
# ```
# This should not be a significant problem due to the logarithm fct slow growth
# %% codecell
# Focusing again on areas:
display('APP areas', df_IsoLog[df_IsoLog['APP_Area'] > 0].corr().loc['APP_Area', seedlings + materials])
display('RL areas', df_IsoLog[df_IsoLog['RL_Area'] > 0].corr().loc['RL_Area', seedlings + materials])
# %% markdown
# More intersting and significant polynomial correlations.
#
# For some reason, the amount of seedlings in an APP area in more correlated to the log of the RL area than with the log of the APP area itself. Something similar happens for the linear correlation.
# %% markdown
# ## Quadractic correlation and perimeter
# %% codecell
df_IsoSqrt = df_Iso.apply(lambda x : np.sqrt(x) if x.name in areas else x)
df_IsoSqrt.rename({'APP_Area':'Sqrt_App_Area', 'RL_Area':'Sqrt_RL_Area', \
    'Total_Area':'Sqrt_Total_Area'}, axis=1, inplace=True)
# %% codecell
df_IsoSqrt[df_IsoSqrt['Sqrt_App_Area'] > 0].corr().loc['Sqrt_App_Area', materials+ seedlings]
correlations_table = pd.DataFrame({
    'APP area':df_Iso[df_Iso['APP_Area'] > 0].corr().loc['APP_Area', materials+ seedlings].round(2),
    'Log APP area':df_IsoLog[df_IsoLog['APP_Area'] > 0].corr().loc['APP_Area', seedlings + materials].round(2),
    'Sqrt APP area':df_IsoSqrt[df_IsoSqrt['Sqrt_App_Area'] > 0].corr().loc['Sqrt_App_Area', materials+ seedlings].round(2),
    'RL area':df_Iso[df_Iso['RL_Area'] > 0].corr().loc['RL_Area', materials + seedlings].round(2),
    'Log RL area':df_IsoLog[df_IsoLog['RL_Area'] > 0].corr().loc['RL_Area', seedlings + materials].round(2),
    'Sqrt RL area':df_IsoSqrt[df_IsoSqrt['Sqrt_RL_Area'] > 0].corr().loc['Sqrt_RL_Area', materials + seedlings].round(2)
    })
display(correlations_table)
# %% markdown
# The above table shows the correlation coefficient between each material's value and the area, its logarithim and its square root, for both APP and RL.
# %% markdown
# We see that all the above materials and seedlings values present a stronger correlation with the perimeter than the area for APP areas, whereas the opposite is true for RL areas. Correlations are still considerably weak but removing zero-area points has significantly incresed them.
#
# As any further perimeter estimation would present a linear relation with the sqrt of the area. Althoug the value itself would differ from the sqrt of area, the correlation coef with the materials and seedlings would remain unchanged.
# %% markdown
# ## Land use discrimination
# %% codecell
dfAct = {}
for area in ['APP_Area', 'RL_Area']:
    materials_activity = {}
    for activity in activities.values():
        materials_activity[activity] = \
        df_Iso[(df_Iso[area] > 0) & \
        ((df_Iso['Main_Activity'] == activity) | (df_Iso['Secondary_Activity'] == activity))]\
        .corr().loc[area, materials+seedlings].round(2)
    dfAct[area] = pd.DataFrame(materials_activity)

display('APP area', dfAct['APP_Area'], 'RL area', dfAct['RL_Area'])
# %% markdown
# The above tables show correlations coefficients between areas and materials (rows' index), for each activity presented in the farm. That is, the number 0.22 at row 1, column 1 of the first table means that among dairy cattle farms, the correlation between number of Piles used and the APP area equals 0.22 approximately.

# %% codecell
activity_categories = {
                'Dairy cattle':'Cattle',
                'Beef cattle':'Cattle',
                'Mixed cattle':'Cattle',
                'Coffee cultivation':'Agriculture',
                'Fruit growing':'Agriculture',
                'Poultry':'Animal farm',
                'Pork':'Animal farm'
                # 'Soap production'
                # 'Agroindustry',
                # 'Lease',
                # 'Horticulture'
                # 'Agriculture',
                # 'Fish farming',
        }

df_Iso[['Main_Activity_Category', 'Secondary_Activity_Category']] =\
 df_Iso[['Main_Activity', 'Secondary_Activity']].replace(activity_categories)
# display(df_Iso[['Main_Activity_Category', 'Secondary_Activity_Category']]_

dfAct_Cat = {}
for area in ['APP_Area', 'RL_Area']:
    materials_activity = {}
    for category in df_Iso['Main_Activity_Category'].unique():
        materials_activity[category] = df_Iso[(df_Iso[area] > 0) & (df_Iso['Main_Activity_Category'] == category)].corr().loc[area, materials+seedlings].round(2)
    dfAct_Cat[area] = pd.DataFrame(materials_activity)
display('APP_Area', dfAct_Cat['APP_Area'], 'RL_Area', dfAct_Cat['RL_Area'])

# %% markdown
# The use of categories to reduce the list of activities into a smaller group
# shows a significant increase in the correlations between materials and area
# for each type of main activity. In particular, we can list
# - APP_Area and Carbonate, Seedlings_APP for Agriculture as Main_Activity_Category
# - RL_Area and almost every material and Seedlings_RL for Agriculture as Main_Activity_Category
#
# However, we notive we are working with a very reduced dataset:

# %% codecell
display(df_Iso[(df_Iso['RL_Area'] > 0) & (df_Iso['Main_Activity_Category'] == 'Agriculture')].shape[0])
display(df_Iso[(df_Iso['APP_Area'] > 0) & (df_Iso['Main_Activity_Category'] == 'Agriculture')].shape[0])

# %% markdown
# O the other hand, there are far more areas whose isolation works haven't been
# labeld as 'completed' ('CONCLUIDO'):

# %% codecell
display(df[(df['Isolation'] != 'CONCLUIDO') & (df['Concluded_Isolation'] == 0)].shape[0])
display(df_Iso[(df_Iso['RL_Area'] > 0)].shape[0])
display(df_Iso[(df_Iso['APP_Area'] > 0)].shape[0])
display(df.shape[0])
display(df_Iso.shape[0])

# %% markdown
# ## Next:
#
# Questions to answer and problems to solve:
#
# - Is the land type (RL, APP) determined by features not displayed above (topography, soil type, p.H. etc.) due to exploitation/cultivation rules associated to each one of these legal labels?
# **Ask Felipe** who choses between RL and APP, Rioterra or the producer
# - Odd correlations between apparently unrelated materials (wire and carbonate, for instance) may indicate hidden features highly correlated to the variables we'll try to predict.
# -  We have a problem: If each line in the dataframe corresponds to a unique reforested poligon, and given that each poligon can be either RL or APP but not both, then how come there are lines with both 'APP_Area' and 'RL_Area' non-null values? In fact,
# ```
# display((df_Iso['APP_Area'][df_Iso['RL_Area'] > 0] == 0).sum())
# ```
# returns 37.
# - Is it possible that areas where the isolation works have been concluded
# haven't been labeled as such in `df['Isolation']` or `df[Concluded_Isolation]`?
# **Ask Felipe**
#
# ### Prediction model:
#
# Given the insights we've gained so far, we can start thinking about the
# predictions model pipeline we want to build as an activities classification
# (forest?) followed by a linear regression using areas and other land features
# we may obtain.
#
# ### Priorities:
#
#  1. Get more data: are there more isolated areas in the dataframe than we know?
#  1. Explore further correlations using land use discrimination: repeat the
# above analysis checking for polynomial correlations and the sqrt of the area
# as before
#  1. Explore further correlations between materials and other variables we
# already have, such as year, location and method, for instance.
#  1.Search for more land features via satelite images (topography)

# %% codecell
df.columns


# for area in ['APP_Area', 'RL_Area']:
#     materials_activity = {}
#     for activity in activity_categories.values():
#         materials_activity[activity] = \
#         df_Iso[(df_Iso[area] > 0) & \
#         ((df_Iso['Main_Activity'] == activity) | (df_Iso['Secondary_Activity'] == activity))]\
#         .corr().loc[area, materials+seedlings].round(2)
#     dfAct[area] = pd.DataFrame(materials_activity)
#
# display('APP area', dfAct['APP_Area'], 'RL area', dfAct['RL_Area'])
