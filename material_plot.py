# # material usage over time

# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import seaborn as sns

# # load the data
# data = pd.read_csv("/users/charles/desktop/pytorch-examples-tutorials/historical-material-data.csv")

# # print(data.head())

# data["Date"] = pd.to_datetime(data["Date"])

# # plot material usage over time
# sns.lineplot(data = data, x="Date", y="Quantity", hue="Material_Type")
# plt.title("Material usage over time")
# plt.show()



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


