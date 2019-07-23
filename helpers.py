import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import json


zips2fips = json.load(open('zip2fips.json','r')) 
zips2fips = dict(zips2fips) 
#zips2fips = defaultdict(int,zips2fips)

def plot_weather_feature_values(feature,daily_weather_df):
    vals = daily_weather_df[feature].values

    #Break protein values into low, med, high based on lowest and highest 10/90th percentile
    N = len(vals)
    sorted_vals = sorted(vals)
    low_vals = sorted_vals[0:int(N*0.05)]
    med_vals = sorted_vals[int(N*0.05):int(N*0.95)]
    high_vals = sorted_vals[int(N*0.95):]

    plt.plot(sorted_vals,np.linspace(0,1,len(vals)))
    plt.axvline(med_vals[0],linestyle='--',alpha=0.7)
    plt.axvline(med_vals[-1],linestyle='--',alpha=0.7)

    plt.xlabel('{} value'.format(feature));
    plt.ylabel('percent')
    plt.title('CDF of {} values'.format(feature));


def plot_target_values_cdf(target_feature,wheat_df):
    vals = wheat_df[target_feature].values

    #Break protein values into low, med, high based on lowest and highest 10/90th percentile
    N = len(vals)
    sorted_vals = sorted(vals)
    low_vals = sorted_vals[0:int(N*0.05)]
    med_vals = sorted_vals[int(N*0.05):int(N*0.95)]
    high_vals = sorted_vals[int(N*0.95):]

    plt.plot(sorted_vals,np.linspace(0,1,len(vals)))
    plt.axvline(med_vals[0],linestyle='--',alpha=0.7)
    plt.axvline(med_vals[-1],linestyle='--',alpha=0.7)

    plt.xlabel('{} value'.format(target_feature));
    plt.ylabel('percent')
    plt.title('CDF of {} values'.format(target_feature));



def plot_correlations(target_feature,weather_features,wheat_df,daily_weather_df):
    #Plot correlations between weather features and low/med/high protein
    for feat in weather_features:
        fig,ax = plt.subplots(1,2,figsize=(15,5))
        for i in range(wheat_df.shape[0]):
            cfips = wheat_df.iloc[i]['fips']
            y = wheat_2014_df.iloc[i][target_feature]
            X = daily_weather_df[daily_weather_df['adm2_code'] == 'US{}'.format(cfips)]
            if X.shape[0] == 0:
                #print('No weather data for fips: {}'.format(cfips))
                continue
            #x = X[feat].mean()
            ax[0].scatter(X[feat].mean(),y)
            ax[1].scatter(X[feat].median(),y)
            
        ax[0].set_title('Average {} vs {} value for 2014 wheat'.format(feat,target_feature));
        ax[1].set_title('Median {} vs {} value for 2014 wheat'.format(feat,target_feature));



def get_monthly_weather_features(wheat_year_df,daily_weather_df,year,weather_features):
    weather_month_features = ['{}_{}'.format(feat,mo) for feat in weather_features for mo in range(12)]
    for f in weather_month_features:
        wheat_year_df[f] = [np.nan for i in range(wheat_year_df.shape[0])]

    daily_weather_year_df = daily_weather_df[daily_weather_df['year'] == year]
    #A lot of fips don't have any weather data, so we won't be able to use those samples in our correlation analysis
    wheat_fips = set(wheat_year_df['fips'].unique())
    weather_fips = {f.strip('US') for f in daily_weather_year_df['adm2_code'].unique()}

    overlap_fips = wheat_fips.intersection(weather_fips)
    #all_fips = set(wheat_year_df['fips'])

    for cfips in overlap_fips:
        X = daily_weather_year_df[daily_weather_year_df['adm2_code'].astype(str) == 'US{}'.format(cfips)]
        for feat in weather_features:
            x = X[feat].values
            for i in wheat_year_df[wheat_year_df['fips'] == cfips].index:
                for mo in range(11):
                    wheat_year_df.loc[i,'{}_{}'.format(feat,mo)] = x[mo*30:(mo+1)*30].mean()
                wheat_year_df.loc[i,'{}_{}'.format(feat,11)] = x[11*30:].mean()

    #wheat_year_df = wheat_year_df[all(~pd.isnull(wheat_year_df))]
    wheat_year_df = wheat_year_df.dropna()

    return wheat_year_df


def process_wheat_data(wheat_data_df):
    wheat_data_df = wheat_data_df[~pd.isnull(wheat_data_df['zip_code'])]
    wheat_data_df['year'] = wheat_data_df['year'].astype(int)
    def convert_zip_to_fips(x):
        if x in zips2fips.keys():
            return zips2fips[x]
        else:
            return np.nan

    wheat_data_df['fips'] = wheat_data_df['zip_code'].map(lambda xx: convert_zip_to_fips(str(int(xx))))
    #remove columns we don't care about
    wheat_data_df = wheat_data_df[['year','zip_code','protein_12','actual_wheat_ash','falling_no','fips']]
    wheat_data_df = wheat_data_df.dropna()
    return wheat_data_df




