import pandas as pd
import matplotlib.pyplot as plt

# Load and filter data
url = "https://data.ssb.no/api/v0/dataset/49626.csv?lang=en"
data = pd.read_csv(url)
data = data[(data['region'] == '0 The whole country') & (data['contents'] == 'Population 1 January')]
data = data[['year', data.columns[-1]]].rename(columns={data.columns[-1]: 'population'})

# Plot the data
plt.plot(data['year'], data['population'], label='Population', color='b')
plt.title('Population Over Years in Norway')
plt.xlabel('Year'); plt.ylabel('Population')
plt.legend(); plt.grid()
plt.show()
