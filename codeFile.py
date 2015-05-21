
import numpy as np
import pandas
import scipy
import scipy.stats
import matplotlib.pyplot as plt


# Load the dataset into pandas dataframe
dataframe = pandas.read_csv("dataset.csv")


# Run Mann Whitney U test
def mann_whitney_plus_means(dataframe):

    rainEntries = dataframe[dataframe['rain'] == 1]
    noRainEntries = dataframe[dataframe['rain'] == 0]
    
    with_rain_mean = np.mean(rainEntries['ENTRIESn_hourly'])
    without_rain_mean = np.mean(noRainEntries['ENTRIESn_hourly'])
    
    result = scipy.stats.mannwhitneyu(rainEntries['ENTRIESn_hourly'], noRainEntries['ENTRIESn_hourly'])
    
    U = result[0]
    p = result[1]
    
    print "The result from Mann Whitney U Test:","Mean of rain is", with_rain_mean, ",   Mean of no rain is", without_rain_mean, ",   The Mann-Whitney statistic is", U, ",   One-sided p-value is", p


# Count number of Heavy Rain  
def countHeavyRain(dataframe):
    heavyRain = dataframe['conds']=='Heavy Rain'
    heavyRainCount = 0
    for i in heavyRain:
        if i == True:
            heavyRainCount += 1
    print "Number of Heavy Rain is" , heavyRainCount


# Count number of storm
def countStorm(dataframe):
    storm = dataframe['conds']=='storm'
    stormCount = 0
    for i in storm:
        if i == True:
            stormCount += 1
    print "Number of Storm is" , stormCount
    

# Normalize the features in the data set.
def normalize_features(array):
    
   array_normalized = (array-array.mean())/array.std()
   mu = array.mean()
   sigma = array.std()
 
   return array_normalized, mu, sigma


# Compute the cost function given a set of features / values, and the values for thetas.
def compute_cost(features, values, theta):
    
    m = len(values)
    sum_of_square_errors = np.square(np.dot(features, theta) - values).sum()
    cost = sum_of_square_errors / (2*m)

    return cost
    
    
# Perform gradient descent     
def gradient_descent(features, values, theta, alpha, num_iterations):
    
    m = len(values)
    cost_history = []

    for i in range(num_iterations):
      
        predictedValue = np.dot(features, theta)
        theta = theta - alpha/m * np.dot((predictedValue - values), features)
        
        #calculate cost
        costresult = compute_cost(features, values, theta)
        
        #appendcost
        cost_history.append(costresult)
        
    return theta, pandas.Series(cost_history)


# Compute R squared
def compute_r_squared(data, predictions):
    
    averageofY = np.mean(data)
    sumofyifi = np.sum((data - predictions)**2)
    sumofyiybar = np.sum((data - averageofY)**2)
    
    r_squared = 1 - sumofyifi/sumofyiybar
    print "R squared value is", r_squared
    return r_squared
    


def predictions(dataframe):

    # Additional features can be added here
    features = dataframe[[]]

   
    # Add conds to features using dummy variables
    dummy_conds = pandas.get_dummies(dataframe['conds'], prefix='c')
    features = features.join(dummy_conds)    

    # Add UNIT to features using dummy variables
    dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix='unit')
    features = features.join(dummy_units)
    
    # Add day_week to features using dummy variables
    dummy_day_week = pandas.get_dummies(dataframe['day_week'], prefix='day_week')
    features = features.join(dummy_day_week)
    
    # Add hour to features using dummy variables
    dummy_hour = pandas.get_dummies(dataframe['hour'], prefix='hour')
    features = features.join(dummy_hour)
    
    # Values
    values = dataframe['ENTRIESn_hourly']
    m = len(values)

    features, mu, sigma = normalize_features(features)
    
    # Add a column of 1s (y intercept)
    features['ones'] = np.ones(m)
   
    # Convert features and values to numpy arrays
    features_array = np.array(features)
    values_array = np.array(values)

    # Set values for alpha, number of iterations.
    alpha = 0.1 # please feel free to change this value
    num_iterations = 75 # please feel free to change this value

    # Initialize theta, perform gradient descent
    theta_gradient_descent = np.zeros(len(features.columns))
    theta_gradient_descent, cost_history = gradient_descent(features_array, 
                                                            values_array, 
                                                            theta_gradient_descent, 
                                                            alpha, 
                                                            num_iterations)
    
    
    # Compute R squared
    predictions = np.dot(features_array, theta_gradient_descent)
    compute_r_squared(values, predictions);




# Display figures
def displayFigures(dataframe):
    
    # Figure one: Histogram of ENTRIESn_hourly with Frequency(05/01/11-05/31/11)
    plt.figure()
    ax = dataframe['ENTRIESn_hourly'][dataframe['rain'] == 0].plot(kind = 'hist', title = 'ENTRIESn_hourly with Frequency(05/01/11-05/31/11)', bins = 30, color = 'green')
    dataframe['ENTRIESn_hourly'][dataframe['rain'] == 1].plot(kind = 'hist', bins = 30, color = 'blue', ax=ax)
    ax.set_ylabel('Frequency')
    ax.set_xlabel('ENTRIESn_hourly')
    ax.legend(['No rain time', 'Rain time'])

    # Figure two: Bar graph of Number of Entries by Day of week(05/01/11-05/31/11)
    plt.figure()
    newdataframe = dataframe.groupby('day_week', as_index=False).sum()
    ax = newdataframe['ENTRIESn_hourly'].plot(kind = 'bar', color = 'green', title = 'Number of Entries by Day of week(05/01/11-05/31/11)')
    ax.set_ylabel('Number of Entries')
    ax.set_xlabel('Day of week')
    ax.legend(['Sum of Entries per day'])
    


mann_whitney_plus_means(dataframe);
countHeavyRain(dataframe);
countStorm(dataframe);
predictions(dataframe);
displayFigures(dataframe);
