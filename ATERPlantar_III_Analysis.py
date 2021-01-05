import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from functools import reduce

# =========================================================================== #
# =========================================================================== #

df = pd.read_csv('Base_ATER.csv', index_col=0)
# df.info()
# df[['COD_PLA', 'Isolation', 'Concluded_Isolation']]\
#     [(df['Isolation'] == 'CONCLUIDO') & (df['Concluded_Isolation'] == 0)]

# SELECT AREAS WITH COMPLETED ISOLATION WORKS ONLY
df_Iso = df[(df['Isolation'] == 'CONCLUIDO') | (df['Concluded_Isolation'] == 1)].reset_index(drop=True)
# df_Iso.head()
df_Iso.drop(['COD_PLA', 'Num_Visits', 'GLEBA', 'LINHA', 'KM', 'LOTE'], axis=1, inplace=True)
df_Iso['Concluded_Isolation'] = df_Iso.loc[:, 'Concluded_Isolation'].astype('category')
# df_Iso.info()

# null_itens = map(lambda x : df_Iso[x].isna().sum(), ['Piles', 'Columns', 'Tensioners', 'Wire_rolls'])
# print(list(null_itens))
# zero_itens = map(lambda x : df_Iso[df_Iso[x] == 0].shape[0], ['Piles', 'Columns', 'Tensioners', 'Wire_rolls'])
# print(list(zero_itens))

materials = ['Piles', 'Columns', 'Tensioners', 'Wire_rolls', 'Carbonate']
seedlings = ['Seedlings_APP','Seedlings_RL']
areas = ['APP_Area', 'RL_Area', 'Total_Area']
df_Iso[materials] = df_Iso[materials].fillna(0).astype(float)
df_Iso[areas] = df_Iso[areas].fillna(0).astype(float)
df_Iso = df_Iso[df_Iso['Total_Area'] > 0]
df_Iso.reset_index(drop=True, inplace=True)
# df_Iso['Total_Area'].value_counts()

# Translations
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
        'OLERICULTURA':'Olericulture'
        }
df_Iso[['Main_Activity', 'Secondary_Activity']] = df_Iso[['Main_Activity', 'Secondary_Activity']].replace(activities)
df_Iso['Secondary_Activity'].fillna('None', inplace=True)

df_Iso.info()
# =========================================================================== #

# Linear correlations:
display(df_Iso.corr().loc[:, materials])
# INTERESTING RESULT:
# correlations between different materials, as expected
# Carbonate corr to nothing
# Largest correlation with areas:
#  - APP_Area and Piles (0.40)
#  - APP_Area and Wire_rolls (0.39)
# Why isn't the same tue for RL? Is the land type (RL, APP) determined by
# features not displayed above (topography, soil type, p.H. etc.) due to
# their exploitation rules?
# Significant correlation between seedlings and Piles, Wire_rolls
# Questions: how is a fence built? how is the amount of seedlings estimated?

display(df_Iso.corr().loc[areas, ['Seedlings_RL', 'Seedlings_APP'] + materials])
# As the amount of seelings and fence materials are poorly related to the area
# (at least w.r.t. linear corr.), it seems that other land features, such as
#  topography, may be very important to such estimations. In other words,
# it seems that there might be hidden variables relating the features we see in

# the df and the materials and seedlings employed, such as topography I suppose.
#
# - Can we access the topography from satelite pictures? Checking on that
#
# The plots below depict fence materials vs. total area. However, as the
# correlation table above indicates, RL and APP areas have very different
# patterns, so we better consider them separately.

plt.scatter(y='Piles', x='Total_Area', data=df_Iso, alpha=0.6)
# plt.scatter(y='Piles', x='RL_Area', data=df_Iso, color='r', alpha=0.6)
# plt.scatter(y='Piles', x='APP_Area', data=df_Iso, color='g', alpha=0.4)
plt.xlabel('Area')
plt.ylabel('Piles')
plt.title('Piles vs Area')
plt.show()


# Materials per type of land use
fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(10,10))
fig.subplots_adjust(wspace=0.5, hspace=0.3, left=0.125, right=0.8, top=1, bottom=0.1)
fig.suptitle('Materials per area (x label)')
ax[0,0].scatter(y='Columns', x='APP_Area', data=df_Iso, color='r')
ax[0,0].set_title('Columns vs. APP area')
ax[0,1].scatter(y='Columns', x='RL_Area', data=df_Iso, color='b')
ax[0,1].set_title('Columns vs. RL area')
ax[1,0].scatter(y='Piles', x='APP_Area', data=df_Iso, color='r')
ax[1,0].set_title('Piles vs APP area')
ax[1,1].scatter(y='Piles', x='RL_Area', data=df_Iso, color='b')
ax[1,1].set_title('Piles vs. RL area')
ax[2,0].scatter(y='Wire_rolls', x='APP_Area', data=df_Iso, color='r')
ax[2,0].set_title('Wire_rolls vs APP Area')
ax[2,1].scatter(y='Wire_rolls', x='RL_Area', data=df_Iso, color='b')
ax[2,1].set_title('Wire_rolls vs RL Area')
ax[3,0].scatter(y='Tensioners', x='APP_Area', data=df_Iso, color='r')
ax[3,0].set_title('Tensioners vs APP Area')
ax[3,1].scatter(y='Tensioners', x='RL_Area', data=df_Iso, color='b')
ax[3,1].set_title('Tensioners vs RL Area')
ax[4,0].scatter(y='Carbonate', x='APP_Area', data=df_Iso, color='r')
ax[4,0].set_title('Carbonate vs APP Area')
ax[4,1].scatter(y='Carbonate', x='RL_Area', data=df_Iso, color='b')
ax[4,1].set_title('Carbonate vs RL Area')
plt.show()
# As the materials values listened are the sums of those employed in both RL and
# APP areas, we see a high peak of material usage in null areas at the far left
# of each plot, particularly for RL areas. This is probably affecting the
# correlation analysis and thus we should separate such cases.

display(df_Iso[df_Iso['APP_Area'] > 0].corr().loc['APP_Area', materials+['Seedlings_APP']])
display(df_Iso[df_Iso['RL_Area'] > 0].corr().loc['RL_Area', materials+['Seedlings_RL']])
# Separating the materials in the areas increase significantly the correlations
# as we exclude the left peak of points in the plots.
#
# We thus conclude that any analysis of Total_Area is of little or no meaning,
# and RL and APP areas ought to be considered separately.


# POLYNOMIAL CORRELATIONS:
df_IsoLog = df_Iso.apply(lambda x : np.log(x + 0.00001) if x.name in materials+seedlings else x)
# Added 0.00001 because I got a divided by zero error message:
#
# 679: RuntimeWarning: divide by zero encountered in log
# result = getattr(ufunc, method)(*inputs, **kwargs)
#
# This should not be a significant problem due to the logarithm fct slow growth
#
display(df_IsoLog.corr().loc[:, materials])
# The fact that apparently unrelated features s.a. Carbonate
# and Wire_rolls may indicate hiden features more strongly connected to each one
# of those. However, it may be that what I'm calling significant correlation may
# be actualy too weak for such conclusion.
#
# Focusing again on areas:
display(df_IsoLog[df_IsoLog['APP_Area'] > 0].corr().loc['APP_Area', seedlings + materials])

display(df_IsoLog[df_IsoLog['RL_Area'] > 0].corr().loc['RL_Area', seedlings + materials])
# No signmificant polynomial correlation


# QUADRACTIC CORRELATION AND PERIMETER
df_IsoSqrt = df_Iso.apply(lambda x : np.sqrt(x) if x.name in areas else x)
df_IsoSqrt.rename({'APP_Area':'Sqrt_App_Area', 'RL_Area':'Sqrt_RL_Area', \
    'Total_Area':'Sqrt_Total_Area'}, axis=1, inplace=True)
# df_IsoSqrt.columns
display(df_IsoSqrt[df_IsoSqrt['Sqrt_App_Area'] > 0].corr().loc['Sqrt_App_Area', materials+ seedlings])

display(df_IsoSqrt[df_IsoSqrt['Sqrt_RL_Area'] > 0].corr().loc['Sqrt_RL_Area', materials + seedlings])

df_IsoSqrt[df_IsoSqrt['Sqrt_App_Area'] > 0].corr().loc['Sqrt_App_Area', materials+ seedlings]
correlations_table = pd.DataFrame({
    'APP area':df_Iso[df_Iso['APP_Area'] > 0].corr().loc['APP_Area', materials+ seedlings].round(2),
    'Sqrt APP area':df_IsoSqrt[df_IsoSqrt['Sqrt_App_Area'] > 0].corr().loc['Sqrt_App_Area', materials+ seedlings].round(2),
    'RL area':df_Iso[df_Iso['RL_Area'] > 0].corr().loc['RL_Area', materials + seedlings].round(2),
    'Sqrt RL area':df_IsoSqrt[df_IsoSqrt['Sqrt_RL_Area'] > 0].corr().loc['Sqrt_RL_Area', materials + seedlings].round(2)
    })
display(correlations_table)
# We see that all the above materials and seedlings values (strangely including)
#  carbonate present a stronger correlation with the perimeter than the area for
# APP areas, whereas the opposite is true for RL areas. Correlations are still
# considerably weak but removing zero-area points has considerably incresed them.
#
# As any further perimeter estimation would present a linear relation with the
# sqrt of the area, althoug the value itself would differ from the sqrt of area
# the correlation coef with the materials and seedlings would remain unchanged.

# ======================================================================== #
# LAND USE

# df_Iso['Main_Activity'].unique()
g = sns.scatterplot(y='Columns', x='APP_Area', data=df_Iso[df_Iso['APP_Area'] > 0], hue='Main_Activity')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

# We have a problem:
# If each line in the dataframe corresponds to a unique reforested poligon, and
# given that each poligon can be either RL or APP but not both, then how come
# there are lines with both 'APP_Area' and 'RL_Area' non-null values?
display((df_Iso['APP_Area'][df_Iso['RL_Area'] > 0] == 0).sum())


dfAct = {}
for area in ['APP_Area', 'RL_Area']:
    materials_activity = {}
    for activity in activities.values():
        materials_activity[activity] = \
        df_Iso[(df_Iso[area] > 0) & \
        ((df_Iso['Main_Activity'] == activity) | (df_Iso['Secondary_Activity'] == activity))]\
        .corr().loc[area, materials+seedlings].round(2)
    dfAct[area] = pd.DataFrame(materials_activity)

display(dfAct['APP_Area'], dfAct['RL_Area'])
# Particularly significant correlations (>0.6):
# For APP areas:
# - Fish farming
# - Fruit growing and Piles, Columns, Wire_rows, Seedlings_RL
# - Horticulture and iles, Columns
# - Agroindustry and Seedlings_APP
# - Lease and Piles, Columns, Wire_rows and Carbonate
# - Olericulture and Piles, Columns, Wire_rows, Carbonate, Seedlings_APP
#
# For RL areas:
# - Beef cattle and Wire_rows
# - Mixed cattle and everything except seedlings variables
# - Coffee and Piles, Wire_rows, Seedlings_RL
# - Agriculture and Columns, Tensioners (but not the other fence materials!) and seedlings
# - Fish farming and Piles, Columns, Tensioners, Wire_rows (why fish farming require so much material??)
#  - Fruit growing and Piles, Tensioners (negative), Wire_rows

df_Iso[(df_Iso['APP_Area'] > 0) & (df_Iso['Main_Activity'] == 'Beef cattle')].corr().loc['APP_Area', materials+seedlings]
pd.DataFrame(materials_activity)
# ============================================================================ #
# Focusing on area:
display(df_Iso.corr().loc[areas, materials])
display(df_IsoLog.corr().loc[areas, materials])
display(df_IsoSqrt.corr().loc[['Sqrt_App_Area', 'Sqrt_RL_Area', 'Sqrt_Total_Area'], materials])
# Largest correlations between areas and materials yet:
#  - APP_Area and Piles (0.42)
#  - APP_Area and Wire_rolls (0.42)
# We notice a very different behaviour for RL
plt.scatter(y='Piles', x='Sqrt_App_Area', data=df_IsoSqrt, alpha=0.6)
# plt.scatter(y='Piles', x='RL_Area', data=df_Iso, color='r', alpha=0.6)
# plt.scatter(y='Piles', x='APP_Area', data=df_Iso, color='g', alpha=0.4)
plt.xlabel('Sqrt of Area')
plt.ylabel('Piles')
plt.title('Piles vs Sqrt Area')
plt.show()

# As using the square root of the are means approximating the area by a square,
# perhaps we can estimate the perimeter directly.

# Estimate permimeter
(df_Iso['APP_Area'] == 0).sum()
df_Iso['Permimeter_APP1'] = df_Iso['APP_Area'].apply(lambda x: 2*np.sqrt(0.1 * x) * (1.1)/0.1)
df_Iso['Permimeter_APP9'] = df_Iso['APP_Area'].apply(lambda x: 2*np.sqrt(0.9 * x) * (1.9)/0.9)
df_Iso[df_Iso['APP_Area'] > 0].corr().loc[['Permimeter_APP1', 'Permimeter_APP9'], materials]
# Changing the shaope of the rectangle produces no great advantage:
# Notice that already the sart values have the same correlation. This comes from
# the way Pearson's r is calculated.
# The correlation value increases significantly (almost 0.1) when we remove
# areas of null surface




df_IsoLog[['Permimeter_APP3', 'Permimeter_RL3', 'Permimeter_Total3']] = \
df_Iso[['Permimeter_APP3', 'Permimeter_RL3', 'Permimeter_Total3']]\
.apply(lambda x: np.log(x+0.00001))
df_IsoLog.corr().loc[['Permimeter_APP3', 'Permimeter_RL3', 'Permimeter_Total3'], materials]

(df_Iso['Total_Area'] == 0).sum()

# CHECK FOR CORRELATION BETWEEN AREAS AND SEEDLINGS IN BOTH CATEGORIES APP AND RL



plt.scatter(y='Columns', x='APP_Area', data=df_Iso, color='r')
plt.xlabel('APP Area')
plt.ylabel('Columns')
plt.title('Columns vs Area')
plt.show()


plt.scatter(y='Columns', x='RL_Area', data=df_Iso, color='g')
plt.xlabel('RL Area')
plt.ylabel('Columns')
plt.title('Columns vs Area')
plt.show()

plt.scatter(y='Tensioners', x='Total_Area', data=df_Iso)
plt.xlabel('Area')
plt.ylabel('Tensioners')
plt.title('Tensioners vs Area')
plt.show()

plt.scatter(y='Wire_rolls', x='Total_Area', data=df_Iso)
plt.xlabel('Area')
plt.ylabel('Wire_rolls')
plt.title('Wire_rolls vs Area')
plt.show()
