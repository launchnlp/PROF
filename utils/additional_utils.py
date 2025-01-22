import numpy as np  
import scipy.stats as stats  
  
# Sample data (replace with your own data)  
results = np.array([4.07, 4, 3.3, 4.1]) / 5.0
  
# Calculate mean for each random seed  
means = np.mean(results)  
  
# Calculate standard deviation for each random seed  
stds = np.std(results)  
  
# Set confidence level (e.g., 95%)  
confidence_level = 0.95  
  
# Calculate critical value based on confidence level  
critical_value = stats.t.ppf((1 + confidence_level) / 2, len(results) - 1)  
  
# Calculate margin of error  
margin_of_error = critical_value * stds / np.sqrt(len(results))
print(margin_of_error)