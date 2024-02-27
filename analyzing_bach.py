import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# def get_bwv_numbers(url):
#     response = requests.get(url)
#     if response.status_code != 200:
#         print("Failed to retrieve the webpage")
#         return []

#     soup = BeautifulSoup(response.content, 'html.parser')
#     links = soup.find_all("a", href=True)
#     bwv_numbers = set()


#     for link in links:
#         href = link['href']
#         match = re.search(r'(\d+)\.html', href)
#         if match:
#             bwv_numbers.add(match.group(1))
        
#     return list(bwv_numbers)

def get_bwv_data(url):
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to retrieve the webpage")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    font_tags = soup.find_all("font", {"face": "Verdana,Arial,Helvetica", "size": "1"})
    
    bwv_data = []

    for font_tag in font_tags:
        links = font_tag.find_all('a')
        if links:
            bwv_number = links[0].get_text(strip=True)
            bwv_title = links[1].get_text(strip=True)
            bwv_link = links[0]['href']
            bwv_data.append((bwv_number, bwv_title, bwv_link))
    
    return bwv_data

# Using the function
index_url = "http://www.bachcentral.com/BWV/index.html"
bwv_data = get_bwv_data(index_url)

print(bwv_data)


# using the function
# index_url = "http://www.bachcentral.com/BWV/index.html"

# bwvs = get_bwv_numbers(index_url)
# print(bwvs)

col_names = ["Title", "Subtitle_and_notes", "BWV", "BWV_epifix", "CLC_BWV_W_epifix", "belongs_after",
            "voices_instruments", "category1", "category2", "category3", "cantate_cat1", "cantate_cat2"]

num_rows = len(bwv_data)

scraped_data = pd.DataFrame(index=range(num_rows), columns=col_names)

# display the dataframe structure
# print(scraped_data)

# first title entry
# first_title_entry = scraped_data['Title'][0]

# print(first_title_entry)

# function to clean and split text
def clean_and_split(text):
    return text.replace('\t', '').split('\n')

# # # iterate over bwv numbers and scrape data
# for i, bwv in enumerate(bwvs):
#     print(f"scraping data for bwv {bwv}")

#     url = f"http://www.bachcentral.com/BWV/{bwv}.html"
#     response = requests.get(url)
#     if response.status_code != 200:
#         print(f"Failed to retrieve data for BWV {bwv}")
#         continue


#     soup = BeautifulSoup(response.content, "html.parser")
#     uls = soup.find_all("ul")

#     for j, ul in enumerate(uls):
#         if j < len(col_names):
#             text = ul.get_text()
#             values = clean_and_split(text)
#             if values:
#                 scraped_data.at[i, col_names[j]] = values[0]


# scraped_data.columns = [col.lower().replace(' ', '_') for col in scraped_data.columns]


# def scrape_bwv_data(bwv):
#     print(f"Scraping data for BWV {bwv}")
#     url = f"http://www.bachcentral.com/BWV/{bwv}.html"
#     response = requests.get(url)

#     if response.status_code != 200:
#         print(f"Failed to retrieve data for BWV {bwv}")
#         return bwv, None

#     soup = BeautifulSoup(response.content, 'html.parser')
#     uls = soup.find_all('ul')

#     data_row = {}
#     for j, ul in enumerate(uls):
#         if j < len(col_names):
#             text = ul.get_text()
#             values = clean_and_split(text)
#             if values:
#                 data_row[col_names[j]] = values[0]

#     return bwv, data_row

# # Using ThreadPoolExecutor to scrape data concurrently
# with ThreadPoolExecutor(max_workers=5) as executor:
#     future_to_bwv = {executor.submit(scrape_bwv_data, bwv): bwv for bwv in bwv_data}
    
#     for future in as_completed(future_to_bwv):
#         bwv = future_to_bwv[future]
#         data_row = future.result()[1]
#         if data_row:
#             index = bwv_data.index(bwv)
#             for col_name, value in data_row.items():
#                 scraped_data.at[index, col_name] = value

# scraped_data.columns = [col.lower().replace(' ', '_') for col in scraped_data.columns]

# display the dataframe


def scrape_bwv_page(url):
    response = requests.get(url)
    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.content, 'html.parser')

    # Example of scraping logic (this needs to be adjusted to the actual page structure)
    title = soup.find("ul").text
    subtitle = soup.find("some_selector_for_subtitle").text
    # ... fetch other details similarly

    return {
        "Title": title,
        "Subtitle_and_notes": subtitle,
        # ... populate other fields
    }

def fetch_data(bwv_entry):
    number, title, link = bwv_entry
    # Prepend the base URL to the relative URL
    full_url = "http://www.bachcentral.com/BWV/" + link
    details = scrape_bwv_page(full_url)
    if details:
        details["Title"] = title
        details["subtitle_and_notes"] = subtitle_and_notes
        details["BWV"] = number
        details["CLC_BWV_W_epifix"] = CLC_BWV_W_epifix
        details["belongs_after"] = belongs_after
        details["voices_instruments"] = voices_instruments
        details["category1"] = category1
        details["category2"] = category2
        details["category3"] = category3
        details["cantate_cat1"] = cantate_cat1
        details["cantate_cat2"] = cantate_cat2
        return details
    return None


# col_names = ["Title", "Subtitle_and_notes", "BWV", "BWV_epifix", "CLC_BWV_W_epifix", "belongs_after",
#             "voices_instruments", "category1", "category2", "category3", "cantate_cat1", "cantate_cat2"]

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(fetch_data, bwv) for bwv in bwv_data]

    for i, future in enumerate(as_completed(futures)):
        result = future.result()
        if result:
            scraped_data.loc[i] = result

print(scraped_data)

# print(scraped_data)