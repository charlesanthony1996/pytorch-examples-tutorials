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
