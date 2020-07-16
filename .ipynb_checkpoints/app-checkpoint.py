## Import all modules
# gunicornpip
# Flask
from flask import Flask, request, render_template, redirect, url_for, jsonify
from wtforms import Form, validators, TextField, SelectField, DecimalField, SubmitField, IntegerField
from wtforms.validators import NumberRange
from flask_wtf import FlaskForm

# General
import numpy as np
import pandas as pd
import random, time, scipy, os, json
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


data = pd.read_csv('static/data/all_city_data_clean.csv').dropna(subset=['City']).reset_index(drop=True)

col = ['State', 'City', 'Total Population', '% Male',
       'Employed Population %', 'Age of the Population',
       '% of people married',
       'Population % with Bachelor Degree or Higher',
       'Median Family Income', '% Below Poverty Level',
       'Average Commute Time',
       'Single People', 'Median Gross Rent', 'Median House Value',
       'Time Zone',
       'Approximate Latitude', 'Approximate Longitude', 'Annual Precip',
       'Summer High', 'Winter Low']

data['Employed Population %'] = data['Employed Population 16+'] / data['Total Population']
data = data.rename(index=str, columns={"People Living Alone": "Single People", 'Male Share of the Population':'% Male'})
data = data[col]

def prepare_data(data, user_input):
    """
    Get rid of NANs to use in nearest neighbors
    Standardize for use later on
    reset indices to match dataset for location at end
    """
    drop_list = ['Time Zone','State','City','Approximate Latitude', 'Approximate Longitude']
    y = pd.DataFrame(user_input.drop(drop_list).dropna()).T
    x = data[y.columns].dropna()

    scaler = StandardScaler()
    x = pd.DataFrame(scaler.fit_transform(x), columns = x.columns).set_index(x.index)
    y = pd.DataFrame(scaler.transform(y), columns = y.columns).set_index(y.index)
    return x, y

def Sort(sub_li):
    """
    This is to sort the distances into increasing order
    to get the shortest n distances
    """
    return(sorted(sub_li, key = lambda x: x[0]))     

def nearest_neighbors(data, obs, orignal_data, n = 5, fake_data = False, original_obs = None):
    """
    data: This is a scaled version of the dataset
    obd: One row DataFrame that with the same scale and columns as data
    n: number of observations 
    
    ------ nearest_neighbors ----
    Takes an average of three distance measures to find the closest point to user input
    
    """
    if fake_data:  
        
        euclidean = scipy.spatial.distance.cdist(data, obs, metric='euclidean')
        manhattan = scipy.spatial.distance.cdist(data, obs, metric='cityblock')
        chebyshev = scipy.spatial.distance.cdist(data, obs, metric='chebyshev')

        combined = (euclidean + manhattan + chebyshev) / 3     
        indices = [ data[ combined == Sort( combined )[i] ].index for i in range(n) ]

        df = orignal_data.loc[ [ str( index[0] ) for index in indices ], : ]
        column_order = df.columns
        df = pd.concat([original_obs, df[:]])
        
        df = df[column_order]
    
    else:
        
        euclidean = scipy.spatial.distance.cdist(data, obs, metric='euclidean')
        manhattan = scipy.spatial.distance.cdist(data, obs, metric='cityblock')
        chebyshev = scipy.spatial.distance.cdist(data, obs, metric='chebyshev')

        combined = (euclidean + manhattan + chebyshev) / 3     
        indices = [ data[ combined == Sort( combined )[i] ].index for i in range(n) ]

        df = orignal_data.loc[ [ str( index[0] ) for index in indices ], : ]
        df.rename(index={'0':'value'}, inplace = True)
          
    return df


# Init the flask 
server = Flask(__name__,
    template_folder="templates",
    static_folder="static")

SECRET_KEY = os.urandom(32)
server.config['SECRET_KEY'] = SECRET_KEY

data_columns = ['Total Population','% Male','Employed Population %','Age of the Population','% of people married','Population % with Bachelor Degree or Higher',
'Median Family Income','% Below Poverty Level','Average Commute Time','Single People','Median Gross Rent',
'Median House Value','Annual Precip','Summer High','Winter Low']

# form for the show entry
class Form(FlaskForm):
    state = SelectField('State:',choices=[(i, i) for i in list(data['State'].unique())]) ##validators=[validators.DataRequired()]
    city = SelectField('City', choices=[], validators=[validators.DataRequired()])
    
    #'Total Population',
    population =  IntegerField('Total Population', default = int(data['Total Population'].mean()), validators=[validators.DataRequired()])
    #'% Male',
    male_per = DecimalField('Male % of Pop', default = data['% Male'].mean()*100)
    #'Employed Population %',
    employed_pop_perc = DecimalField('Employed Population %', default = data['Employed Population %'].mean()*100)
    #'Age of the Population',
    age = IntegerField('Average Age', default = int(data['Age of the Population'].mean()))
    #'% of people married',
    married_perc = DecimalField('Married % of Pop', default = data['% of people married'].mean()*100)
    #'Population % with Bachelor Degree or Higher',
    bach_deg_perc = DecimalField('Pop % with Bachelor Degree +', default = data['Population % with Bachelor Degree or Higher'].mean()*100)
    #'Median Family Income',
    income = IntegerField('Median Income', default = int(data['Median Family Income'].mean()))
    #'% Below Poverty Level',
    below_pov_perc = DecimalField('Pop % Below Poverty', default = data['% Below Poverty Level'].mean()*100)
    #'Average Commute Time',
    commute = IntegerField('Average Commute Time (Mins)', default = int(data['Average Commute Time'].mean()))
    #'Single People',
    singles = DecimalField('Pop % That Live Single', default = data['Single People'].mean()*100)
    #'Median Gross Rent',
    rent = IntegerField('Average Rent', default = int(data['Median Gross Rent'].mean()))
    #'Median House Value',
    home_value = IntegerField('Average Home Value', default = int(data['Median House Value'].mean()))
    #'Annual Precip',
    precip = DecimalField('Annual Precipitaion', default = int(data['Annual Precip'].mean()))
    #'Summer High',
    high = IntegerField('Average Annual High Temp', default = int(data['Summer High'].mean()))
    #'Winter Low'
    low = IntegerField('Average Annual Low Temp', default = int(data['Winter Low'].mean()))  
    
@server.route('/', methods=['GET', 'POST'])
def index():        
    
    form = Form()    
    alaska_cities = data[data['State']=='Alabama']['City']
    form.city.choices = [ (city, index) for city, index in zip(alaska_cities.index, alaska_cities) ]
    
    if request.method == 'POST':
        
        ## State Select List
        if 'state_search' in request.form:
            
            user_choice = data.loc[ str(form.city.data), : ]
            x, y = prepare_data(data, user_choice)
            df = nearest_neighbors(data=x, obs=y, orignal_data=data, n = 6)
            
            trans_data = df.T
            trans_data.columns = ['Your Choice','Most Similar','2nd','3rd','4th','5th']
            
            format_dict = {'Total Population': '{:,}', '% Male': '{:.2%}', 'Employed Population %': '{:.2%}', '% of people married': '{:.2%}',
               'Population % with Bachelor Degree or Higher':'{:.2%}', 'Median Family Income':'${0:,.0f}','% Below Poverty Level': '{:.2%}', 'Single People': '{:.2%}',
               'Median Gross Rent':'${0:,.0f}', 'Median House Value':'${0:,.0f}', 'Annual Precip': '{:.1f}'}
            
            return render_template('output.html', data=trans_data.to_html(classes='table table-hover table-dark', header="true"), df=df)   
        
        # User Choice of Ideal City Features
        elif 'user_form' in request.form:
            
            fields_data = {'name':[], 'value':[]}
            for field in form:
                if field.name not in ['state','city','csrf_token']:
                    if field.data != 9999999999:
                        fields_data['name'].append(field.name)
                        fields_data['value'].append(field.data)
                    else:
                        fields_data['name'].append(field.name)
                        fields_data['value'].append(np.nan)
                        
            user_choice = pd.DataFrame(fields_data).T
            user_choice.columns = user_choice.iloc[0]
            user_choice = user_choice[1:]
            
            rename_dict = {'population':'Total Population','male_per':'% Male','employed_pop_perc':'Employed Population %','age':'Age of the Population','married_perc':'% of people married',
                           'bach_deg_perc':'Population % with Bachelor Degree or Higher','income':'Median Family Income','below_pov_perc':'% Below Poverty Level','commute':'Average Commute Time',
                           'singles':'Single People','rent':'Median Gross Rent','home_value':'Median House Value','precip':'Annual Precip','high':'Summer High','low':'Winter Low'}
            user_choice = user_choice.rename(columns = rename_dict)
            
            fill_in_df = pd.DataFrame({'State':'Ideal State', 'City':'Ideal City','Time Zone':'No Time Zone','Approximate Latitude':999,'Approximate Longitude':999}, index=['value'])
            user_choice = user_choice.join(fill_in_df)
            
            for col in user_choice.columns:
                if col in ['% Male', 'Employed Population %','% of people married','Population % with Bachelor Degree or Higher','% Below Poverty Level', 'Single People']:
                    user_choice[col] = user_choice[col] / 100
            
            x, y = prepare_data(data, user_choice.T)
            df = nearest_neighbors(data=x, obs=y, orignal_data=data, n = 5, fake_data=True, original_obs = user_choice)
            
            trans_data = df.T
            trans_data.columns = ['Your Choice','Most Similar','2nd','3rd','4th','5th']
            
            return render_template('output.html', data=trans_data.to_html(classes='table table-hover table-dark', header="true"), df=df)        
        
    return render_template('index.html', form = form,
                           states = list(data['State'].unique()),
                           cities = list(data['City'].unique()))

@server.route('/about', methods=['GET', 'POST'])
def about_page():  
    return render_template('about.html')

@server.route('/contact', methods=['GET', 'POST'])
def contact_page():  
    return render_template('contact.html')

cities = {}
for state in list(data['State'].unique()):
    cities[state] = list(data[data['State']==state]['City'].unique())

@server.route('/city/<state>')
def city(state):
    cities = data[data['State']==state]['City']
    
    city_array = []
    for city, index in zip(cities, cities.index):
        cityobj = {'index':index, 'name':city}
        city_array.append(cityobj)
    
    return jsonify({'cities':city_array})

if __name__ == '__main__':
    server.run(use_reloader = False, debug=True)