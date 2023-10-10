# # material usage over time

# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import plotly.express as px

# # load the data
# data = pd.read_csv("/users/charles/desktop/pytorch-examples-tutorials/historical-material-data.csv")

# # print(data.head())

# data["Date"] = pd.to_datetime(data["Date"])

# # plot material usage over time
# # sns.lineplot(data = data, x="Date", y="Quantity", hue="Material_Type")
# # plt.title("Material usage over time")
# # plt.show()

# # using plotly's api to add the features
# fig = px.line(data, x="Date", y="Quantity", color="Material_Type", title="Material usage over time", 
#             labels={"Quantity": "Usage_Quantity", "Date": "Date", "Material_Type":"Material_Type"}, 
#             template="plotly_dark")

# fig.update_xaxes(rangeslider_visible=True)

# # add annotations (example: annotate maximum value, can be customized further)
# max_point = data[data["Quantity"] == data["Quantity"].max()]
# fig.add_annotation(x=max_point["Date"].values[0], y= max_point["Quantity"].values[0], text="Maximum", showarrow =True, arrowhead= 1)

# fig.show()

# ===========================

# heatmap of material and product type

# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import seaborn as sns

# # load the data 
# df = pd.read_csv("/users/charles/desktop/pytorch-examples-tutorials/historical-material-data.csv")

# # print(df.tail())

# heatmap_data = df.groupby(["Material_Type", "Product_Type"]).size().unstack()
# sns.heatmap(heatmap_data, annot=True ,cmap="YlGnBu")
# plt.title("Material vs product type")
# plt.show()


# heatmap of material and product type using plotly
# import pandas as pd
# import plotly.figure_factory as ff

# # load the data
# df = pd.read_csv("/users/charles/desktop/pytorch-examples-tutorials/historical-material-data.csv")

# # heatmap data
# heatmap_data = df.groupby(["Material_Type", "Product_Type"]).size().unstack(fill_value = 0)

# # create a heatmap using plotly
# fig = ff.create_annotated_heatmap(
#     z= heatmap_data.values,
#     x= heatmap_data.columns.tolist(),
#     y= heatmap_data.index.tolist(),
#     colorscale="Viridis",
#     showscale=True,
#     annotation_text=heatmap_data.values
# )

# # update layout
# fig.update_layout(title="Material vs Product type", xaxis_title="Product Type", yaxis_title="Material Type")

# fig.show()


# =========================


# histogram of bend angles (assuming bend angles are comma seperated)
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import seaborn as sns

# df = pd.read_csv("/users/charles/desktop/pytorch-examples-tutorials/historical-material-data.csv")

# # print(df.info())

# # expanding the angles column
# angles = data["Angles"].str.split(",", expand=True).stack()
# sns.histplot(angles, bins=30)
# plt.title("Histogram of bend angles")
# plt.show()


# import pandas as pd
# import plotly.express as px

# # sample data loading (adjust the path as needed)
# df = pd.read_csv("/users/charles/desktop/pytorch-examples-tutorials/historical-material-data.csv")

# print(df.head())

# # expanding the angles column
# angles = df["Angles"].str.split(",", expand=True).stack().astype(float).reset_index(drop=True)

# # create histogram using plotly
# fig = px.histogram(angles,nbins=30, title="Histogram of bend angles", labels={'value': 'Bend Angles', 'count': 'Frequency'},
#  color_discrete_sequence=['#636EFA'], marginal="box")

# fig.add_shape(type="line", x0=angles.median(), x1=angles.median(), y0=0, y1=1, yref="paper",
#  line=dict(color="green", width= 2, dash="dash"), name="Median")

# fig.update_layout(xaxis_title="Bend angles(degrees)", yaxis_title="Frequency", legend_title="Statistics", hovermode="closest")

# fig.show()



# ==============================



# scatter plot for thickness vs quantity (proxy for wastage)
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import pandas as pd

# df = pd.read_csv("/users/charles/desktop/pytorch-examples-tutorials/historical-material-data.csv")

# print(df.head())

# sns.scatterplot(data = data, x="Thickness_mm", y="Quantity", hue="Material_Type")
# plt.title("Thickness vs Quantity")
# plt.show()


import pandas as pd
import plotly.express as px

# load the data (adjust the path as needed)
df = pd.read_csv("/users/charles/desktop/pytorch-examples-tutorials/historical-material-data.csv")

# create scatter plot using plotly
fig = px.scatter(df,x="Thickness_mm", y="Quantity", color="Material_Type", title="Thickness vs Quantity",
 labels={"Thickness_mm": "Thickness (mm)", 'Quantity': 'Quantity ordered'}, hover_data=["Material_Type"])


# add reference lines if needed (average thickness and average quantity)
avg_thickness = df["Thickness_mm"].mean()
avg_quantity = df["Quantity"].mean()

fig.add_shape(type="line", x0=avg_thickness, x1=avg_thickness, y0=0, y1=df["Quantity"].max(),
line=dict(color="Red", width=2, dash="dash"))

fig.add_shape(type="line", x0=0, x1=df["Thickness_mm"].max(), y0=avg_quantity, y1=avg_quantity,
line=dict(color="Blue", width=2, dash="dash"))

fig.update_layout(hovermode="closest", template="plotly_dark")

# fig.show()
