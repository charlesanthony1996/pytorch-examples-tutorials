# sql -> select * from regions

import pandas as pd
employees = pd.read_csv(r"employees.csv")
departments = pd.read_csv(r"departments.csv")
job_history = pd.read_csv(r"job_history.csv")
jobs = pd.read_csv(r"jobs.csv")
countries = pd.read_csv(r"countries.csv")
regions = pd.read_csv(r"regions.csv")
locations = pd.read_csv(r"locations.csv")

# print("all the records from the regions file")
# print(regions)

# result = locations[['LOCATION_ID']]
# print(result)

# extract first 7 records of the employees file
# print(employees.head(7))

