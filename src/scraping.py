import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
from datetime import datetime

headers = {
    "User-Agent": "Mozilla/5.0"
}


def get_sp500_companies():

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url, headers=headers)
    tables = pd.read_html(response.text)
    sp500 = tables[0]
    tickers = sp500["Symbol"].tolist()
    print("Empresas S&P500:", len(tickers))
    return tickers


def scrape_reuters(company, max_news=3):

    url = f"https://www.reuters.com/site-search/?query={company}"
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.find_all("a")
    news = []

    for a in articles:
        title = a.text.strip()
        link = a.get("href")
        if title and link and "/markets/" in link:
            news.append({
                "date": datetime.now(),
                "company": company,
                "headline": title,
                "url": "https://www.reuters.com" + link
            })

        if len(news) >= max_news:
            break
    return news


def run_scraper():
    tickers = get_sp500_companies()
    all_news = []
    # solo 30 empresas
    selected_companies = tickers[:30]
    for company in selected_companies:
        print("\nBuscando noticias:", company)
        news = scrape_reuters(company, max_news=3)
        print("Noticias encontradas:", len(news))
        all_news.extend(news)
        time.sleep(1)
    df = pd.DataFrame(all_news)
    print("\nTotal noticias recolectadas:", len(df))
    df.to_csv("data/raw/news.csv", index=False)
    print("Dataset guardado en data/raw/news.csv")


if __name__ == "__main__":
    run_scraper()