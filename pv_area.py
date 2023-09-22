import geopandas as gpd
import matplotlib.pyplot as plt

# read the shapefile
gdf = gpd.read_file("/users/charles/desktop/pytorch-examples-tutorials/PHOTOVOLTAIK_GUENSTIGE_BEREICHE/PHOTOVOLTAIK_GUENSTIGE_BEREICHE.shp")

# plot the shapefile
# gdf.plot()

plt.title('photovoltaic suitable area')
plt.xlabel('longitude')
plt.ylabel('latitude')

# plt.show()

print(gdf.head())

total_area = gdf["Shape_STAr"].sum()
print(total_area)
print()
# largest and smallest areas
max_area = gdf["Shape_STAr"].max()
min_area = gdf["Shape_STAr"].min()

print()
print(f"Largest area: {max_area}")
print(f"Smallest area: {min_area}")

upper_austria_area = 11718320000

percentage_of_largest_pv_area = (max_area/ 11718320000) * 100

print(percentage_of_largest_pv_area,"%")

# area distribution
gdf["Shape_STAr"].hist()

plt.title("Distribution of suitable area sizes")
plt.xlabel("Area")
plt.ylabel("Number of regions")
# plt.show()



# solar energy production in upper austria using all the available area

# total_area = df['Shape_STAr'].sum()

# constants for energy production calculation
solar_panel_eff = 0.18
solar_irradiance = 1000
performance_ratio = 0.8

# estimate the amount of energy production
annual_energy_production = total_area * solar_irradiance * solar_panel_eff * performance_ratio * 24 * 365.25

print(f"total suitable energy area: {total_area:.2f} m^2")
print(f"estimated annual energy production: {annual_energy_production:.2f} kwh/year")


# find out about how much energy upper austria needs per year


# spatial density
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt


# extract the coords
gdf["x"] = gdf.geometry.centroid.x
gdf["y"] = gdf.geometry.centroid.y

# plotting
fig, ax = plt.subplots(figsize=(6, 6))
sns.kdeplot(x=gdf['x'], y = gdf["y"] , fill=True, cmap="viridis", ax=ax, cbar=True)

ax.set_title("density of suitable areas for photovoltaic installations")
ax.set_xlabel("longitude")
ax.set_ylabel("latitude")

# plt.show()

# scatter plot for area and perimeter

# extracting areas and parameters
areas = gdf["Shape_STAr"]
perimeter = gdf["Shape_STLe"]

# creating the scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(areas, perimeter, alpha= 0.6)
plt.title("Area vs perimeterof suitable zones")
plt.xlabel("Area (m^2)")
plt.ylabel("Perimeter (m)")
plt.grid(True)
# plt.show()



# how much does energy does upper austria consume per year?

# assuming that annual_energy_production is the variable from your previous code
# that holds the estimated energy production from the pv installation

# replace this with the actual energy consumption data you found
upper_austria_energy_consumption = 320 * 10**15

annual_energy_production_joules = annual_energy_production * 3.6e6

percentage_covered = (annual_energy_production_joules  / upper_austria_energy_consumption) * 100
print(f"the potential pohotovoltaic installations could cover approximately {percentage_covered:.2f}% of upper austrias annual energy consumption")

import torch
import torch.nn as nn

class TimeSeriesPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size = 1):
        super(TimeSeriesPredictor, self).__init__()
        self.rnn = nn.RNN(input_size , hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.linear(x[-1])
        return x


# example of data (replace with your time-series data)

data = torch.randn(10, 5, 1)

model = TimeSeriesPredictor(1, 50)
output = model(data)

# print(output)
