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

# select distinct id of the employees file
# result = employees.DEPARTMENT_ID.unique()
# print(result)


# display name, department number for all employees whose last name is mcewen
# result = employees[employees.LAST_NAME == "McEwen"]
# for index, row in result.iterrows():
#     print(row['LAST_NAME'],'  ',row['FIRST_NAME'],' ',row['DEPARTMENT_ID'])

# display the name, salary, and department number
# result = employees[employees['FIRST_NAME'].str[:1] == 'S']
# for index, row in result.iterrows():
#     print(row['FIRST_NAME'].ljust(15), row['LAST_NAME'].ljust(15),str(row['SALARY']).ljust(9),row['DEPARTMENT_ID'])


# display the name, salary and department number for those employees whose first name does not contain the letter
# 'M'
result = employees[employees['FIRST_NAME'].str.find('M') == -1]
# print(result)
for index, row in result.iterrows():
    print(row['LAST_NAME'].ljust(15), row['FIRST_NAME'].ljust(15))

