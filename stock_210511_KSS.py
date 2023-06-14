# -*- coding: utf-8 -*-
"""
Created on Tue May 11 21:25:50 2021

@author: USER
"""

# 주가정보를 크롤링해 주식 분석

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stock_crawler import StockCrawler

crawler = StockCrawler('A066570')       # LG전자의 주식 코드
filename = crawler.get_prices(5)

df = pd.read_csv(filename, encoding='utf-8')
# print(df.head()); print()

# # 존체 데이터 구성 확인
# print(df.describe()); print()

# #날짜 시간 형식으로 변경, 인덱스로 만든다, 책의 656, 657 페이지 참조
df['날짜'] = pd.to_datetime(df['날짜'], format="%Y-%m-%d", errors='coerce')
# print(df['날짜']); print()
df.set_index('날짜', inplace=True)        # 원본 변경
# print(df.head()); print()

df.sort_index(inplace=True)             # 날짜를 오름차순으로 재정리, 과거 --> 현재

# 한글 폰트
from matplotlib import font_manager
[(f.name, f.fname) for f in font_manager.fontManager.ttflist]
plt.rc('font', family='LG Smart UI')

# 주식 그래프
plt.figure(dpi=120)
plt.plot(df.index, df["저가"], color="blue", label="저가")
plt.plot(df.index, df["고가"], color="red", label="고가")
plt.legend(loc='best')
plt.xlabel("날짜")
plt.ylabel("주가")
plt.title("LGE Stock Graph")
plt.grid(True)

# 이동 평균 구하기
df['5일이평'] = df['종가'].rolling(window=5).mean()
df['30일이평'] = df['종가'].rolling(window=30).mean()
df['어제5일이평'] = df['5일이평'].shift(1)
df['어제30일이평'] = df['30일이평'].shift(1)

# 골든 크로스
# 어제 단기 < 장기, 오늘 단기 > 장기
df['golden'] = (df['5일이평'] > df['30일이평']) & (df['어제5일이평'] < df['어제30일이평'])
# print(df[df['golden']==True])

# 데드 크로스
# 어제 단기 > 장기, 오늘 단기 < 장기
df['dead'] =(df['5일이평'] < df['30일이평']) & (df['어제5일이평'] > df['어제30일이평'])
# print(df[df['dead']==True])

# # 단기 이평 1가지, 장기 이평 1가지 그리기
plt.figure(dpi=120)
plt.plot(df.index, df['5일이평'], label='MA5')
plt.plot(df.index, df['30일이평'], label='MA30')
plt.plot(df[df.golden==True].index,
        df['5일이평'][df.golden==True],
        "^",
        c='gold',
        label='Golden'
        )
plt.plot(df[df.dead==True].index,
        df['5일이평'][df.dead==True],
        "v",
        c='black',
        label='Dead'
        )
plt.legend(loc='best')
plt.xlabel("날짜")
plt.xticks(fontsize=10, rotation=30)
plt.ylabel("주가")
plt.title("MA Graph")
plt.grid(True)
plt.tight_layout()
plt.savefig("stock_test.png")

