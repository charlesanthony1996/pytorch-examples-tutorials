import time
import csv
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

df = pd.read_csv("annual-co2-emissions-austria.csv")

# select the last 10 years
df_last_10_years = df.tail(10)


# scatter plot
# plt.scatter(df["Year"], df["Annual CO2 emissions"])

# step plot
# plt.step(df['Year'], df['Annual CO2 emissions'])

# fill between plot
# plt.fill_between(df['Year'], df['Annual CO2 emissions'])

# fill betweenx -> x axis
# plt.fill_betweenx(df['Year'], df['Annual CO2 emissions'])

# stairs plot
# plt.stairs(df['Year'], df['Annual CO2 emissions'])

# basic line plot for all years
# plt.plot(df['Year'], df['Annual CO2 emissions'])

# basic line plot for the last 10 years
plt.plot(df_last_10_years['Year'], df_last_10_years['Annual CO2 emissions'])






plt.title("Annual CO2 emissions in austria")

plt.xlabel("Year")
plt.ylabel("Annual CO2 emissions in MTPC")

# the background part
fig = plt.gcf()
fig.patch.set_facecolor("white")

# create a rectangle
x = df_last_10_years['Year'].iloc[5]
y = df_last_10_years['Annual CO2 emissions'].iloc[5]

width = 4
height = df_last_10_years['Annual CO2 emissions'].iloc[6] - df_last_10_years['Annual CO2 emissions'].iloc[5]

rect = Rectangle((x, y), width, height, linewidth= 1, edgecolor="black", facecolor="none")
# plt.gca().add_patch(rect)

# plt.show()