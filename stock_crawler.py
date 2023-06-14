#from requests_html import HTMLSession
# pip install requests_html
import requests
from bs4 import BeautifulSoup
import csv
import json
class StockCrawler:
    def __init__(self, code):
        self.url_format = "http://finance.daum.net/api/quote/%s/days?symbolCode=%s&page=%d&perPage=100&pagination=true"
        self.code = code

    def get_prices(self,page=50):
        csvfile = open(self.code + ".csv", "w", encoding='utf-8', newline='')
        filename = csvfile.name
        stock_writer = csv.writer(csvfile)
        stock_writer.writerow(["날짜", "시가", "고가", "저가", "종가"])

        custom_headers = {
            "user-agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.87 Safari/537.36",
            "referer":f"http://finance.daum.net/quotes/{self.code}",
        }
        for page in range(1, page + 1):
            url = self.url_format % (self.code, self.code, page)
            data = requests.get(url, headers=custom_headers)
            if data.status_code != requests.codes.ok:
                print("접속실패")

            stock_data = json.loads(data.text)

            for daily in stock_data['data']:
                stock_writer.writerow([daily['date'][:10], daily['openingPrice'], daily['highPrice'],
                     daily['lowPrice'],daily['tradePrice']])

        csvfile.close()
        return filename

# 현재 이 소스코드를 단독 실행했을 때 실행될 코드들
if __name__ == "__main__":
    stock_code = input("수집을 원하는 주식의 코드를 입력하세요 : ")
    crawler = StockCrawler(stock_code)
    filename = crawler.get_prices(5)
