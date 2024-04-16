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

# sql query -> select first_name, last_name, salary, department_id from employees where first_name not like '%M%';
# result = employees[employees['FIRST_NAME'].str.find('M') == -1]
# # print(result)
# for index, row in result.iterrows():
#     print(row['LAST_NAME'].ljust(15), row['FIRST_NAME'].ljust(15))


# display the name, salary and department_id in ascending order by department number
# result = employees.sort_values('DEPARTMENT_ID', ascending=True)
# for index, row in result.iterrows():
#     print(row['FIRST_NAME'])


# display the name, salary and department number
# result = employees.sort_values('FIRST_NAME', ascending=False)
# for index, row in result.iterrows():
#     print(row['FIRST_NAME'].ljust(15))


# create a boolean series, where true for not null and False for null values for
# or missing values in specified column
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)

# print('original data / state province')
# print(locations.STATE_PROVINCE)


# display the name, salary and department number for those employees whose first
# name ends with the letter 'm'
# result = employees[employees['FIRST_NAME'].str[-1] == 'm']
# print(result)

