from bs4 import BeautifulSoup
import requests
import re

# def getHTMLdocument(url):
#     response = requests.get(url)
#     # return response.text
#     return BeautifulSoup(response.text, 'html.parser')

def getHTMLdocument(url, proxies,access_token):
    headers = {
        "Authorization" : f"Bearer {access_token}"
    }
    try:
        response = requests.get(url, headers=headers, proxies=proxies)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the document: {e}")
        return None

# # def getHTMLdocument(url,access_token):
# #     headers = {
# #         "Authorization" : f"Bearer {access_token}"
# #     }
# #     try:
# #         response = requests.get(url, headers=headers, timeout=10)
# #         response.raise_for_status()
# #         return BeautifulSoup(response.text, 'html.parser')
# #     except requests.exceptions.RequestException as e:
# #         print(f"Error fetching in URL: {e}")
# #         if response is not None and hasattr(response, "text"):
# #             print(f"Response status code: {response.status_code}")
# #             print(f"Response content: {response.text}")
# #         return None

# def getHTMLdocument(url,access_token,proxies):
#     session = requests.session()

#     session.headers.update({
#         "Authorization" : f"Bearer {access_token}"
#     })
#     try:
#         response = session.get(url, timeout=10)
#         response.raise_for_status()
#         return BeautifulSoup(response.text, 'html.parser')
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching in URL: {e}")
#         return None
    

url_link = "Your_URL_LINK"
pat = "YOUR_GITHUB_ACCOUNT_TOKEN"
if pat is None:
    print("No token")

# proxies = {    
#     'http_proxy' : 'http://uyw1kor:BGSWWelcome@2024@localhost:3128',
#     'https_proxy' : 'http://uyw1kor:BGSWWelcome@2024@localhost:3128',
#     }
proxies = {    
    'http_proxy' : 'http://localhost:3128',
    'https_proxy' : 'http://localhost:3128',
    }

# Html_data = getHTMLdocument(url_link,pat,proxies)
Html_data = getHTMLdocument(url_link,proxies,pat)

if Html_data:
    print(Html_data.get_text(separator="\n", strip=True))

else:
    print("Failed to fetch the document")






