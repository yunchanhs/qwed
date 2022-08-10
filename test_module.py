import time
import pyupbit
import datetime
import pandas
import logging
import requests
import time
import smtplib
import jwt
import sys
import uuid
import hashlib
import math
import numpy
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from urllib.parse import urlencode
import os
import sys
import logging
import math
import traceback


# 공통 모듈 Import
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from bitcoinAutoTrade import upbit as upbit
  


access = "UxavXwLQLeMi6iEjxb4p8Dy6rwmk9GhzB2l8Dr8I"
secret = "2NPZtGBJ0VV9sPjcvL76kd6N4opwVgpxpj1jUi3E"
server_url = "https://api.upbit.com/v1/market/all"
# 로그인
upbit = pyupbit.Upbit(access, secret)
print("autotrade start")
print(upbit.get_balances())
querystring = {"isDetails":"false"}
coinlist = ["KRW-BTC", "KRW-XRP", "KRW-ETC", "KRW-ETH", "KRW-BCH", "KRW-EOS"]
lower28 = []
higher70 = []


def set_loglevel(level):
    try:
 
        # ---------------------------------------------------------------------
        # 로그레벨 : DEBUG
        # ---------------------------------------------------------------------
        if level.upper() == "D":
            logging.basicConfig(
                format='[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d]:%(message)s',
                datefmt='%Y/%m/%d %I:%M:%S %p',
                level=logging.DEBUG
            )
        # ---------------------------------------------------------------------
        # 로그레벨 : ERROR
        # ---------------------------------------------------------------------
        elif level.upper() == "E":
            logging.basicConfig(
                format='[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d]:%(message)s',
                datefmt='%Y/%m/%d %I:%M:%S %p',
                level=logging.ERROR
            )
        # ---------------------------------------------------------------------
        # 로그레벨 : INFO
        # ---------------------------------------------------------------------
        else:
            # -----------------------------------------------------------------------------
            # 로깅 설정
            # 로그레벨(DEBUG, INFO, WARNING, ERROR, CRITICAL)
            # -----------------------------------------------------------------------------
            logging.basicConfig(
                format='[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d]:%(message)s',
                datefmt='%Y/%m/%d %I:%M:%S %p',
                level=logging.INFO
            )
 
    # ----------------------------------------
    # Exception Raise
    # ----------------------------------------
    except Exception:
        raise
        
 
# -----------------------------------------------------------------------------
# - Name : send_request
# - Desc : 리퀘스트 처리
# - Input
#   1) reqType : 요청 타입
#   2) reqUrl : 요청 URL
#   3) reqParam : 요청 파라메타
#   4) reqHeader : 요청 헤더
# - Output
#   4) reponse : 응답 데이터
# -----------------------------------------------------------------------------
def send_request(reqType, reqUrl, reqParam, reqHeader):
    try:
 
        # 요청 가능회수 확보를 위해 기다리는 시간(초)
        err_sleep_time = 0.3
 
        # 요청에 대한 응답을 받을 때까지 반복 수행
        while True:
 
            # 요청 처리
            response = requests.request(reqType, reqUrl, params=reqParam, headers=reqHeader)
 
            # 요청 가능회수 추출
            if 'Remaining-Req' in response.headers:
 
                hearder_info = response.headers['Remaining-Req']
                start_idx = hearder_info.find("sec=")
                end_idx = len(hearder_info)
                remain_sec = hearder_info[int(start_idx):int(end_idx)].replace('sec=', '')
            else:
                logging.error("헤더 정보 이상")
                logging.error(response.headers)
                break
 
            # 요청 가능회수가 3개 미만이면 요청 가능회수 확보를 위해 일정시간 대기
            if int(remain_sec) < 3:
                logging.debug("요청 가능회수 한도 도달! 남은횟수:" + str(remain_sec))
                time.sleep(err_sleep_time)
 
            # 정상 응답
            if response.status_code == 200 or response.status_code == 201:
                break
            # 요청 가능회수 초과인 경우
            elif response.status_code == 429:
                logging.error("요청 가능회수 초과!:" + str(response.status_code))
                time.sleep(err_sleep_time)
            # 그 외 오류
            else:
                logging.error("기타 에러:" + str(response.status_code))
                logging.error(response.status_code)
                break
 
            # 요청 가능회수 초과 에러 발생시에는 다시 요청
            logging.info("[restRequest] 요청 재처리중...")
 
        return response
 
    # ----------------------------------------
    # Exception Raise
    # ----------------------------------------
    except Exception:
        raise
        
# -----------------------------------------------------------------------------
# - Name : get_candle
# - Desc : 캔들 조회
# - Input
#   1) target_item : 대상 종목
#   2) tick_kind : 캔들 종류 (1, 3, 5, 10, 15, 30, 60, 240 - 분, D-일, W-주, M-월)
#   3) inq_range : 조회 범위
# - Output
#   1) 캔들 정보 배열
# -----------------------------------------------------------------------------
def get_candle(target_item, tick_kind, inq_range):
    try:
 
        # ----------------------------------------
        # Tick 별 호출 URL 설정
        # ----------------------------------------
        # 분붕
        if tick_kind == "1" or tick_kind == "3" or tick_kind == "5" or tick_kind == "10" or tick_kind == "15" or tick_kind == "30" or tick_kind == "60" or tick_kind == "240":
            target_url = "minutes/" + tick_kind
        # 일봉
        elif tick_kind == "D":
            target_url = "days"
        # 주봉
        elif tick_kind == "W":
            target_url = "weeks"
        # 월봉
        elif tick_kind == "M":
            target_url = "months"
        # 잘못된 입력
        else:
            raise Exception("잘못된 틱 종류:" + str(tick_kind))
 
        logging.debug(target_url)
 
        # ----------------------------------------
        # Tick 조회
        # ----------------------------------------
        querystring = {"market": target_item, "count": inq_range}
        res = send_request("GET", server_url + "/v1/candles/" + target_url, querystring, "")
        candle_data = res.json()
 
        logging.debug(candle_data)
 
        return candle_data
 
    # ----------------------------------------
    # Exception Raise
    # ----------------------------------------
    except Exception:
        raise
        
 
# -----------------------------------------------------------------------------
# - Name : get_macd
# - Desc : MACD 조회
# - Input
#   1) target_item : 대상 종목
#   2) tick_kind : 캔들 종류 (1, 3, 5, 10, 15, 30, 60, 240 - 분, D-일, W-주, M-월)
#   3) inq_range : 캔들 조회 범위
#   4) loop_cnt : 지표 반복계산 횟수
# - Output
#   1) MACD 값
# -----------------------------------------------------------------------------
def get_macd(target_item, tick_kind, inq_range, loop_cnt):
    try:
 
        # 캔들 데이터 조회용
        candle_datas = []
 
        # MACD 데이터 리턴용
        macd_list = []
 
        # 캔들 추출
        candle_data = get_candle(target_item, tick_kind, inq_range)
 
        # 조회 횟수별 candle 데이터 조합
        for i in range(0, int(loop_cnt)):
            candle_datas.append(candle_data[i:int(len(candle_data))])
 
        df = pd.DataFrame(candle_datas[0])
        df = df.iloc[::-1]
        df = df['trade_price']
 
        # MACD 계산
        exp1 = df.ewm(span=12, adjust=False).mean()
        exp2 = df.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        exp3 = macd.ewm(span=9, adjust=False).mean()
 
        for i in range(0, int(loop_cnt)):
            macd_list.append(
                {"type": "MACD", "DT": candle_datas[0][i]['candle_date_time_kst'], "MACD": round(macd[i], 4), "SIGNAL": round(exp3[i], 4),
                 "OCL": round(macd[i] - exp3[i], 4)})
 
        return macd_list
 
    # ----------------------------------------
    # 모든 함수의 공통 부분(Exception 처리)
    # ----------------------------------------
    except Exception:
        raise
def rsi(ohlc: pandas.DataFrame, period: int = 14):
    delta = ohlc["close"].diff()
    ups, downs = delta.copy(), delta.copy()
    ups[ups < 0] = 0
    downs[downs > 0] = 0

    AU = ups.ewm(com = period-1, min_periods = period).mean()
    AD = downs.abs().ewm(com = period-1, min_periods = period).mean()
    RS = AU/AD

    return pandas.Series(100 - (100/(1 + RS)), name = "RSI")

if __name__ == '__main__':
 
    # noinspection PyBroadException
    
 
        print("***** USAGE ******")
        print("[1] 로그레벨(D:DEBUG, E:ERROR, 그외:INFO)")
 
        # 로그레벨(D:DEBUG, E:ERROR, 그외:INFO)
        upbit.set_loglevel('I')
 
        # ---------------------------------------------------------------------
        # Logic Start!
        # ---------------------------------------------------------------------
        # MACD 조회
        macd_data = upbit.get_macd('KRW-BTC', '30', '200', 10)
 
        for macd_data_for in macd_data:
            logging.info(macd_data_for)
         
        for i in range(len(coinlist)):
         data = pyupbit.get_ohlcv(ticker=coinlist[i], interval="minute3")
         now_rsi = rsi(data, 14).iloc[-1]
         print("코인명: ", coinlist[i])
         print("현재시간: ", datetime.datetime.now())
         print("RSI :", now_rsi)
         print("macd :", macd_data)
         
         if now_rsi <= 28 : 
            
            lower28[i] = True
         elif now_rsi >= 33 and lower28[i] and macd_data > 0 == True:
            
            buy(coinlist[i])
            lower28[i] = False
         elif now_rsi >= 70 and higher70[i] and macd_data < 0 == False:
        
            sell(coinlist[i])
            higher70[i] = True
         elif now_rsi <= 60 :
            
            higher70[i] = False
         time.sleep(0.8)

    #시장가 매수
         def buy(coin):
           money = upbit.get_balance("KRW")
           if money < 20000 :
             res = upbit.buy_market_order(coin, money)
           elif money < 50000:
             res = upbit.buy_market_order(coin, money*0.4)
           elif money < 100000 :
             res = upbit.buy_market_order(coin, money*0.3)
           else :
             res = upbit.buy_market_order(coin, money*0.2)
             return

         def sell(coin):
           amount = upbit.get_balance(coin)
           cur_price = pyupbit.get_current_price(coin)
           total = amount * cur_price
           if total < 20000 :
             res = upbit.sell_market_order(coin, amount)
           elif total < 50000:
             res = upbit.sell_market_order(coin, amount*0.4)
           elif total < 100000:
             res = upbit.sell_market_order(coin, amount*0.3)        
           else :
             res = upbit.sell_market_order(coin, amount*0.2)
             return
