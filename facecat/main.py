# -*- coding:utf-8 -*-
#! python3
from facecat import *
#这里可能需要pip install requests
import requests
from requests.adapters import HTTPAdapter
import random
from datetime import datetime
from stock import *


import pandas as pd
import sys
from model import Kronos, KronosTokenizer, KronosPredictor
import torch

latestDataStr = ""
# tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base",force_download=True, cache_dir="model/Kronos-Tokenizer-base")
# model = Kronos.from_pretrained("NeoQuasar/Kronos-small",force_download=True, cache_dir="model/Kronos-small")
tokenizer = KronosTokenizer.from_pretrained(
    "model/Kronos-Tokenizer-base"  # 本地分词器路径
)
model = Kronos.from_pretrained(
    "model/Kronos-small"  # 本地模型路径
)
if torch.cuda.is_available():
	device = "cuda:0"
elif torch.backends.mps.is_available():
	device = "mps"
else:
	device = "cpu"
predictor = KronosPredictor(model, tokenizer, device=device, max_context=512, )
latestDataStr = ""
findMyCharts = []
charts = []
currentCode = "600000.SH"
clientTicks = dict()
priceRowMap = dict()

def transToPanda(datas):
	data_list = []
	for data in datas:
		data_list.append({
			'timestamps': datetime.fromtimestamp(data.date),
			'open': data.open,
			'high': data.high,
			'low': data.low,
			'close': data.close,
			'volume': data.volume,
			'amount': data.amount
		})
	df = pd.DataFrame(data_list)
	return df

def transToChartData(pred_df):
    data_list = []
    for index, row in pred_df.iterrows():
        data = SecurityData()
        data.open = row['open']
        data.date = time.mktime(index.timetuple())
        data.high = row['high']
        data.low = row['low']
        data.close = row['close']
        data.volume = row['volume']
        data.amount = row['amount']
        data_list.append(data)
    return data_list

def progress_callback(progress, total):
	progressDiv = findViewByName("progress", gPaint.views)
	progressDiv.text = str(progress/total)
	progressDiv.invalidate()

def predict(chart):
	"""
	使用历史K线数据对未来进行预测
	"""
	preButton = findViewByName("preButton", gPaint.views)
	preButton.enabled = False
	progressDiv = findViewByName("progress", gPaint.views)
	progressDiv.text = "-1"
	progressDiv.invalidate()
	preButton.invalidate()
	lookback = chart.lookback
	pred_len = chart.pred_len
	mode = chart.preMode
	
	if len(chart.datas) < lookback and mode == "predict":
		popupWindow(f"k线数量{len(chart.datas)}小于训练数量{lookback}")
		progressDiv.text = "0"
		progressDiv.invalidate()
		preButton.enabled = True
		preButton.invalidate()
		return
	elif len(chart.datas) < lookback + pred_len and mode == "backtest":
		popupWindow(f"k线数量{len(chart.datas)}小于训练数量{lookback + pred_len}")
		progressDiv.text = "0"
		progressDiv.invalidate()
		preButton.enabled = True
		preButton.invalidate()
		return
	if mode == "predict": # 预测模式
		print(f"Using device: {device} mode: {mode}")
		# 可见k线数量
		klines = chart.lastVisibleIndex - chart.firstVisibleIndex 
		if chart.pred_len > 50 and klines > 50:
			chart.firstVisibleIndex += 50
		elif chart.pred_len <= 50 and klines > chart.pred_len:
			chart.firstVisibleIndex += chart.pred_len
		chart.invalidate()
	# 获取图表数据，转化成panda格式
		df = transToPanda(chart.datas)
		df['timestamps'] = pd.to_datetime(df['timestamps'])

		df = df.sort_values('timestamps').reset_index(drop=True)

		x_df = df.tail(lookback)[['open', 'high', 'low', 'close', 'volume', 'amount']].reset_index(drop=True)
		x_timestamp = df.tail(lookback)['timestamps'].reset_index(drop=True)

		# 3. 生成未来的时间戳 (这里我们假设是连续的未来日期)
		last_timestamp = x_timestamp.iloc[-1]
		# 注意：对于股票市场，应该生成交易日，但为简单起见，我们先生成连续日历日
		y_timestamp = pd.Series(pd.date_range(start=last_timestamp + pd.Timedelta(days=1), periods=pred_len))
		pred_df = predictor.predict(
			df=x_df,
			x_timestamp=x_timestamp,
			y_timestamp=y_timestamp,
			pred_len=pred_len,
			T=chart.temperature,
			top_p=chart.topP,
			sample_count=1,
			verbose=True,
			progress_callback=progress_callback
		)
		chart.datas2 = transToChartData(pred_df)
	elif mode == "backtest": # 回测模式
		# popupWindow()
		print(f"Using device: {device} mode: {mode}")
		df = transToPanda(chart.datas)
		df['timestamps'] = pd.to_datetime(df['timestamps'])

		df = df.sort_values('timestamps').reset_index(drop=True)

		test_df = df.tail(lookback + pred_len).reset_index(drop=True)

		x_df = test_df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
		x_timestamp = test_df.loc[:lookback-1, 'timestamps']
		y_timestamp = test_df.loc[lookback:lookback+pred_len-1, 'timestamps']

		# 4. Make Prediction
		pred_df = predictor.predict(
			df=x_df,
			x_timestamp=x_timestamp,
			y_timestamp=y_timestamp,
			pred_len=pred_len,
			T=chart.temperature,
			top_p=chart.topP,
			sample_count=1,
			verbose=True,
			progress_callback=progress_callback
		)
		print(pred_df)
		
		chart.datas2 = transToChartData(pred_df)
	preButton.enabled = True
	preButton.invalidate()

def predict_in_thread(chart):
	"""
	在独立线程中运行预测，以避免UI阻塞.
	"""
	predict(chart)
	chart.invalidate()

def startHttpRequest(url, callBack, tag):
	"""开始Http请求
	url:地址
	callBack回调"""
	data = FCData()
	data.key = url
	data.callBack = callBack
	try:
		s = requests.Session()
		s.mount('http://', HTTPAdapter(max_retries=3))
		response = s.get(url)
		result = response.text
		data.success = True
		data.data = result
	except requests.exceptions.RequestException as e:
		data.success = False
		data.data = str(e)
	data.tag = tag
	gPaint.addData(data)
	user32.PostMessageW(gPaint.hWnd, 0x0401, 0, 0)

def startQueryNewData():
	"""开始请求最新数据"""
	while gPaint.hWnd != 0:
		time.sleep(3)
		queryNewData()

def startQueryPriceData():
	"""开始请求最新数据"""
	while gPaint.hWnd != 0:
		time.sleep(6)
		queryPrice()

def httpRequest(url, callBack, tag):
	"""进行Http请求
	url:地址
	callBack回调"""
	thread = threading.Thread(target=startHttpRequest, args=(url, callBack, tag))
	thread.start()

def onPaint(view, paint, clipRect):
	"""绘制视图
	view:视图
	paint:绘图对象
	clipRect:区域"""
	if view.viewType == "latestdiv":
		drawLatestDiv(view, paint, clipRect)
	else:
		onPaintDefault(view, paint, clipRect)


def drawMyDiv(view, paint, clipRect):
		# elif view.viewName == "temperatureDiv" or view.viewName == "topPDiv":
	tSize = paint.textSize(view.text, "Default,14")
	paint.drawText(view.text, "rgb(255,255,255)",  "Default,14", 3, (view.size.cy - tSize.cy)/2)
	paint.drawLine("rgb(255,255,255)", 1, 0, view.size.cx-1, 0, view.size.cx-1, view.size.cy)
	
def drawProgressDiv(view, paint, clipRect):
	if view.text == "-1":
		paint.drawText("训练中...", "rgb(255,255,255)", "Default,14", 3, 7)
	else:
		cx = float(view.text) * view.size.cx
		if cx > 0 and cx < view.size.cx:
			paint.fillRect(view.textColor, 0, 0, cx, view.size.cy)

def drawPreChart(view, paint, clipRect):
	drawChartStock(view, paint, clipRect)
	if view.datas2 != None and len(view.datas2) > 0:
		cWidth = int(view.hScalePixel - 3) / 2
		splitIndex = view.lastVisibleIndex + 1
		if view.preMode == "backtest":
			splitIndex = view.lastVisibleIndex - view.pred_len + 1
			if view.lastVisibleIndex + 1 < len(view.datas):
				splitIndex = len(view.datas) - 1 - view.pred_len + 1
		elif view.preMode == "predict":
			if view.lastVisibleIndex + 1 < len(view.datas):
				return
		splitX = getChartX(view, splitIndex) - cWidth
		paint.drawLine("rgb(150,150,150)", 1, 0, splitX, 0, splitX, view.size.cy)
		for i in range(0,len(view.datas2)):
			index = view.lastVisibleIndex + i + 1
			
			if view.preMode == "backtest":
				index = view.lastVisibleIndex - view.pred_len + i + 1
				if view.lastVisibleIndex + 1 < len(view.datas):
					index = len(view.datas) - view.pred_len + i + 1
			x = getChartX(view, index)
			if x > view.size.cx - view.rightVScaleWidth:
				break
			m_open = view.datas2[i].open
			m_close = view.datas2[i].close
			m_high = view.datas2[i].high
			m_low = view.datas2[i].low
			openY = getChartY(view, 0, m_open)
			closeY = getChartY(view, 0, m_close)
			highY = getChartY(view, 0, m_high)
			lowY = getChartY(view, 0, m_low)
			if m_close >= m_open:				
				paint.fillRect(view.upColor2, x, highY, x + view.lineWidth, lowY)
				if cWidth > 0:
					if int(closeY) == int(openY):
						paint.drawLine(view.upColor2, 1, 0, x - cWidth, closeY, x + cWidth, closeY)
					else:
						paint.fillRect(view.upColor2, x - cWidth, closeY, x + cWidth + 1, openY)
			else:
				paint.fillRect(view.downColor2, x, highY, x + view.lineWidth, lowY)
				if cWidth > 0:
					paint.fillRect(view.downColor2, x - cWidth, openY, x + cWidth + 1, closeY)
			volY = getChartY(view, 1, view.datas2[i].volume)
			zeroY = getChartY(view, 1, 0)
			if m_close >= m_open:
				barColor = view.upColor2
				if view.volColor != "none":
					barColor = view.volColor

				if cWidth > 0:
					paint.fillRect(barColor, x - cWidth, volY, x + cWidth + 1, zeroY)
				else:
					paint.drawLine(barColor, view.lineWidth, 0, x - cWidth, volY, x + cWidth, zeroY)
			else:
				barColor = view.downColor2
				if view.volColor != "none":
					barColor = view.volColor
				if cWidth > 0:
					paint.fillRect(barColor, x - cWidth, volY, x + cWidth + 1, zeroY)
				else:
					paint.drawLine(barColor, view.lineWidth, 0, x - cWidth, volY, x + cWidth, zeroY)

def drawUpButton(button, paint, clipRect):
	"""绘制温度按钮
	view:视图
	paint:绘图对象
	clipRect:区域"""
	"""重绘按钮 
	button:视图 
	paint:绘图对象 
	clipRect:裁剪区域"""
	r_left = 0
	r_right = button.size.cx
	r_top = 0
	r_bottom = button.size.cy
	r_width = r_right - r_left
	#常规情况
	if button.backColor != "none":
		apt = []
		apt.append(FCPoint(r_left + r_width/2,r_top))
		apt.append(FCPoint(r_left, r_top + r_width/2))
		apt.append(FCPoint(r_right, r_top + r_width/2))
		paint.fillPolygon(button.textColor, apt)
	#鼠标按下
	if button == paint.touchDownView:
		apt = []
		apt.append(FCPoint(r_left + r_width/2,r_top))
		apt.append(FCPoint(r_left, r_top + r_width/2))
		apt.append(FCPoint(r_right, r_top + r_width/2))
		paint.fillPolygon(button.pushedColor, apt)
	#鼠标悬停
	elif button == paint.touchMoveView:
		apt = []
		apt.append(FCPoint(r_left + r_width/2,r_top))
		apt.append(FCPoint(r_left, r_top + r_width/2))
		apt.append(FCPoint(r_right, r_top + r_width/2))
		paint.fillPolygon(button.hoveredColor, apt)

def drawDownButton(button, paint, clipRect):
	r_left = 0
	r_right = button.size.cx
	r_top = 0
	r_bottom = button.size.cy
	r_width = r_right - r_left
	#常规情况
	if button.backColor != "none":
		apt2 = []
		apt2.append(FCPoint(r_left + r_width/2,r_bottom))
		apt2.append(FCPoint(r_right, r_bottom - r_width/2))
		apt2.append(FCPoint(r_left, r_bottom - r_width/2))
		paint.fillPolygon(button.textColor, apt2)
	#鼠标按下
	if button == paint.touchDownView:
		apt2 = []
		apt2.append(FCPoint(r_left + r_width/2,r_bottom))
		apt2.append(FCPoint(r_right, r_bottom - r_width/2))
		apt2.append(FCPoint(r_left, r_bottom - r_width/2))
		paint.fillPolygon(button.pushedColor, apt2)
	#鼠标悬停
	elif button == paint.touchMoveView:
		apt2 = []
		apt2.append(FCPoint(r_left + r_width/2,r_bottom))
		apt2.append(FCPoint(r_right, r_bottom - r_width/2))
		apt2.append(FCPoint(r_left, r_bottom - r_width/2))
		paint.fillPolygon(button.hoveredColor, apt2)

def drawLatestDiv(view, paint, clipRect):
	"""绘制买卖档
	view:视图
	paint:绘图对象
	clipRect:区域"""
	global latestDataStr
	avgHeight = 20
	drawFont = "Default,14"
	textColor = "rgb(175,196,228)"
	if view.paint.defaultUIStyle == "light":
		textColor = "rgb(0,0,0)"
	dTop = 30
	paint.drawLine(view.borderColor, 1, 0, 0, dTop, view.size.cx, dTop)
	dataStrs = latestDataStr.split(",")
	lastClose = 0
	priceList = []
	volList = []
	buySellTexts = []
	if len(dataStrs) > 10:
		paint.drawText(dataStrs[0], textColor, "Default,14", 5, 7)
		paint.drawText(dataStrs[1], "rgb(194,151,18)", "Default,14", 80, 7)
		lastClose = float(dataStrs[8])
		priceList.append(float(dataStrs[23]))
		priceList.append(float(dataStrs[22]))
		priceList.append(float(dataStrs[21]))
		priceList.append(float(dataStrs[20]))
		priceList.append(float(dataStrs[19]))
		priceList.append(float(dataStrs[9]))
		priceList.append(float(dataStrs[10]))
		priceList.append(float(dataStrs[11]))
		priceList.append(float(dataStrs[12]))
		priceList.append(float(dataStrs[13]))

		volList.append(float(dataStrs[28]))
		volList.append(float(dataStrs[27]))
		volList.append(float(dataStrs[26]))
		volList.append(float(dataStrs[25]))
		volList.append(float(dataStrs[24]))
		volList.append(float(dataStrs[14]))
		volList.append(float(dataStrs[15]))
		volList.append(float(dataStrs[16]))
		volList.append(float(dataStrs[17]))
		volList.append(float(dataStrs[18]))

	buySellTexts.append("卖5")
	buySellTexts.append("卖4")
	buySellTexts.append("卖3")
	buySellTexts.append("卖2")
	buySellTexts.append("卖1")
	buySellTexts.append("买1")
	buySellTexts.append("买2")
	buySellTexts.append("买3")
	buySellTexts.append("买4")
	buySellTexts.append("买5")
	maxVol = maxValue(volList)
	for i in range(0, 10):
		tSize = paint.textSize(buySellTexts[i], drawFont)
		paint.drawText(buySellTexts[i], textColor, drawFont, 5, dTop + avgHeight / 2 - tSize.cy / 2)
		if len(priceList) > 0:
			price = priceList[i]
			upDownColor = "rgb(255,82,82)"
			upDownColor2 = "rgb(50,0,0)"
			if price < lastClose:
				upDownColor = "rgb(46,255,50)"
				upDownColor2 = "rgb(0,50,0)"
				if paint.defaultUIStyle == "light":
					upDownColor = "rgb(0,200,0)"
					upDownColor2 = "rgba(0,200,0,50)"
			paint.drawText(toFixed(priceList[i], 2), upDownColor, drawFont, 50, dTop + avgHeight / 2 - tSize.cy / 2)
			volText = toFixed(volList[i] / 100, 0)
			volTextSize = paint.textSize(volText, drawFont)
			paint.drawText(volText, textColor, drawFont, view.size.cx - volTextSize.cx - 10, dTop + avgHeight / 2 - volTextSize.cy / 2)
		dTop += avgHeight
	paint.drawLine(view.borderColor, 1, 0, 0, dTop, view.size.cx, dTop)
	paint.drawText("现价", textColor, drawFont, 5, dTop + 10)
	paint.drawText("幅度", textColor, drawFont, 5, dTop + 35)
	paint.drawText("总额", textColor, drawFont, 5, dTop + 60)
	paint.drawText("总量", textColor, drawFont, 5, dTop + 85)
	paint.drawText("开盘", textColor, drawFont, 110, dTop + 10)
	paint.drawText("振幅", textColor, drawFont, 110, dTop + 35)
	paint.drawText("最高", textColor, drawFont, 110, dTop + 60)
	paint.drawText("最低", textColor, drawFont, 110, dTop + 85)
	if len(dataStrs) > 10:
		close = float(dataStrs[2])
		high = float(dataStrs[3])
		low = float(dataStrs[4])
		open = float(dataStrs[5])
		volume = float(dataStrs[6])
		amount = float(dataStrs[7])
		diff = 0
		if lastClose > 0:
			diff = 100 * (close - lastClose) / lastClose
		diff2 = 0
		if lastClose > 0:
			diff2 = 100 * (high - lastClose) / lastClose - 100 * (low - lastClose) / lastClose
		paint.drawText(toFixed(close, 2), getPriceColor(close, lastClose), drawFont, 40, dTop + 10)
		paint.drawText(toFixed(diff, 2) + "%", getPriceColor(close, lastClose), drawFont, 40, dTop + 35)
		paint.drawText(toFixed(amount / 10000, 0), textColor, drawFont, 40, dTop + 60)
		paint.drawText(toFixed(volume / 10000, 0), textColor, drawFont, 40, dTop + 85)

		paint.drawText(toFixed(open, 2), getPriceColor(open, lastClose), drawFont, 150, dTop + 10)
		paint.drawText(toFixed(diff2, 2) + "%", getPriceColor(close, lastClose), drawFont, 150, dTop + 35)
		paint.drawText(toFixed(high, 2), getPriceColor(high, lastClose), drawFont, 150, dTop + 60)
		paint.drawText(toFixed(low, 2), getPriceColor(low, lastClose), drawFont, 150, dTop + 85)

def historyDataCallBack(data):
	"""历史数据回调"""
	if data.success:
		code = data.tag[0]
		name = data.tag[1]
		cycle = data.tag[2]
		myCharts = data.tag[3]
		result = data.data
		dataList = []
		strs = result.split("\r\n")
		#分时线处理
		for c in range(0, len(myCharts)):
			myChart = myCharts[c]
			chart = myChart.views[0].secondView.views[0]
			myCycle = int(myChart.exAttributes["cycle"])
			if myCycle == 0:
				if cycle == 0:
					fStrs = strs[0].split(" ")
					if len(fStrs) >= 3:
						chart.firstOpen = float(fStrs[2])
					else:
						chart.firstOpen = 0
			else:
				chart.firstOpen = 0
		for i in range(2, len(strs)):
			subStrs = strs[i].split(",")
			if len(subStrs) >= 7:
				data = SecurityData()
				if cycle < 1440:
					dateStr = subStrs[0] + " " + subStrs[1][0:2] + ":" + subStrs[1][2:4] + ":00"
					data.open = float(subStrs[2])
					data.high = float(subStrs[3])
					data.low = float(subStrs[4])
					data.close = float(subStrs[5])
					data.volume = float(subStrs[6])
					date_obj = datetime.strptime(dateStr, "%Y-%m-%d %H:%M:%S")
					data.date = time.mktime(date_obj.timetuple())
					#分时线处理
					if cycle == 0 and (data.volume > 0 or len(dataList) == 0):
						for c in range(0, len(myCharts)):
							myChart = myCharts[c]
							myCycle = int(myChart.exAttributes["cycle"])
							if myCycle == 0:
								chart = myChart.views[0].secondView.views[0]
								chart.lastValidIndex = len(dataList)
								if chart.firstOpen == 0:
									chart.firstOpen = data.close
				else:
					data.open = float(subStrs[1])
					data.high = float(subStrs[2])
					data.low = float(subStrs[3])
					data.close = float(subStrs[4])
					data.volume = float(subStrs[5])
					dateStr = subStrs[0]
					date_obj = datetime.strptime(dateStr, "%Y-%m-%d")
					data.date = time.mktime(date_obj.timetuple())
				dataList.append(data)
		for c in range(0, len(myCharts)):
			myChart = myCharts[c]
			myCycle = int(myChart.exAttributes["cycle"])
			chart = myChart.views[0].secondView.views[0]
			copyDatas = []
			for d in range(0, len(dataList)):
				copyDatas.append(dataList[d])
			#分时线
			if cycle == 0 and myCycle == 0:
				chart.autoFillHScale = True
				chart.cycle = "trend"
			#分钟线
			elif cycle == 1 and (myCycle > 0 and myCycle < 1440):
				chart.cycle = "minute"
				if myCycle > 1:
					newDatas = []
					multiMinuteSecurityDatas(newDatas, copyDatas, myCycle)
					copyDatas = newDatas
			#日线
			elif cycle == 1440 and myCycle >= 1440:
				chart.cycle = "day"
				if myCycle == 10080:
					newDatas = []
					getHistoryWeekDatas(newDatas, copyDatas)
					copyDatas = newDatas
				elif myCycle == 43200:
					newDatas = []
					getHistoryMonthDatas(newDatas, copyDatas)
					copyDatas = newDatas
				elif myCycle == 129600:
					newDatas = []
					getHistorySeasonDatas(newDatas, copyDatas)
					copyDatas = newDatas
				elif myCycle == 259200:
					newDatas = []
					getHistoryHalfYearDatas(newDatas, copyDatas)
					copyDatas = newDatas
				elif myCycle == 518400:
					newDatas = []
					getHistoryYearDatas(newDatas, copyDatas)
					copyDatas = newDatas
			#不符合的K线
			else:
				continue
			clientTicks[c] = ClientTickDataCache()
			setChartTitle(chart, code, name, myCycle)
			chart.lastVisibleKey = 0
			chart.firstVisibleIndex = -1
			chart.lastVisibleIndex = -1
			chart.datas = copyDatas
			maxVisibleRecord = getChartMaxVisibleCount(chart, chart.hScalePixel, getChartWorkAreaWidth(chart))
			if maxVisibleRecord > len(chart.datas):
				chart.firstVisibleIndex = 0
			else:
				chart.firstVisibleIndex = chart.lastVisibleIndex - maxVisibleRecord + 1
			resetChartVisibleRecord(chart)
			checkChartLastVisibleIndex(chart)
			calcChartIndicator(chart)
			chart.invalidate()

def queryHistoryData(code, name, cycle, myCharts):
	"""请求历史数据"""
	url = "http://www.jjmfc.com:9968/quote?func=getkline&code=" + code +  "&cycle=" + str(cycle) + "&count=5000"
	if cycle == 0:
		url = "http://www.jjmfc.com:9968/quote?func=getkline&code=" + code +  "&cycle=0&count=240"
	tag = []
	tag.append(code)
	tag.append(name)
	tag.append(cycle)
	tag.append(myCharts)
	httpRequest(url, historyDataCallBack, tag)

def newDataCallBack(data):
	"""最新数据回调"""
	if data.success:
		global latestDataStr
		result = data.data
		latestDataStr = result
		dataStrs = latestDataStr.split(",")
		if len(dataStrs) > 0:
			code = dataStrs[0]
			close = float(dataStrs[2])
			volume = float(dataStrs[6])
			amount = float(dataStrs[7])
			dateStr = dataStrs[29].replace("\r\n", "")
			date_obj = datetime.strptime(dateStr, "%Y-%m-%d %H:%M:%S")
			date = time.mktime(date_obj.timetuple())
			for c in range(0, len(findMyCharts)):
				myChart = findMyCharts[c]
				chart = charts[c]
				if chart.viewName=="preChart":
					return
				if code == chart.text.split(" ")[0]:
					if len(chart.datas) > 0:
						myCycle = int(myChart.exAttributes["cycle"])
						latestData = SecurityData()
						latestData.close = close
						latestData.open = latestData.close
						latestData.high = latestData.close
						latestData.low = latestData.close
						latestData.volume = volume
						latestData.amount = amount
						latestData.date = date
						cTick = clientTicks[c]
						if len(cTick.code) == 0:
							cTick.code = currentCode
							cTick.lastAmount = amount
							cTick.lastVolume = volume
							cTick.lastDate = latestData.date
						oldDataSize = len(chart.datas)
						if myCycle == 0:
							for d in range(0, len(chart.datas)):
								if chart.datas[d].volume > 0:
									chart.lastValidIndex = d
						if chart.lastRecordIsVisible:
							newDataSize = len(chart.datas)
							if newDataSize > oldDataSize :
								chart.firstVisibleIndex = chart.firstVisibleIndex + 1
								chart.lastVisibleIndex = chart.lastVisibleIndex + 1
					
						checkChartLastVisibleIndex(chart)
						calcChartIndicator(chart)
		gPaint.update()

def queryNewData():
	"""请求最新数据"""
	url = "http://www.jjmfc.com:9968/quote?func=getnewdata&codes=" + currentCode
	tag = []
	httpRequest(url, newDataCallBack, tag)

def setChartTheme(chart, index):
	"""黑色风格"""
	if chart.paint.defaultUIStyle == "dark":
		chart.backColor = "rgb(0,0,0)"
		chart.borderColor = "none"
		chart.textColor = "rgb(175,196,228)"
		chart.scaleColor = "rgb(75,75,75)"
		chart.crossTipColor = "rgb(50,50,50)"
		chart.crossLineColor = "rgb(100,100,100)"
		chart.gridColor = "rgb(50,50,50)"
		if index > 0:
			chart.upColor = "rgb(186,56,18)"
			chart.downColor = "rgb(31,182,177)"
		else:
			chart.upColor = "rgb(255,82,82)"
			chart.downColor = "rgb(46,255,50)"
		chart.barStyle = "rect2"
		chart.candleStyle = "rect2"
		chart.trendColor = "rgb(255,255,255)"
		chart.hScaleTextColor = "rgb(194,151,18)"
		chart.vScaleTextColor = "rgb(194,151,18)"
	elif chart.paint.defaultUIStyle == "light":
		chart.backColor = "rgb(255,255,255)"
		chart.borderColor = "none"
		chart.textColor = "rgb(0,0,0)"
		chart.scaleColor = "rgb(175,175,175)"
		chart.crossTipColor = "rgb(200,200,200)"
		chart.crossLineColor = "rgb(150,150,150)"
		chart.gridColor = "rgb(200,200,200)"
		chart.trendColor = "rgb(50,50,50)"

def setChartTitle(chart, code, name, intCycle):
	"""设置图表的标题"""
	chart.text = code + " " + name
	if intCycle == 0:
		chart.text += " 分时"
	elif intCycle < 1440:
		chart.text += " " + str(intCycle) + "分钟"
	elif intCycle == 1440:
		chart.text += " 日线"
	elif intCycle == 10080:
		chart.text += " 周线"
	elif intCycle == 43200:
		chart.text += " 月线"
	elif intCycle == 129600:
		chart.text += " 季线"
	elif intCycle == 259200:
		chart.text += " 半年线"
	elif intCycle == 518400:
		chart.text += " 年线"

def getPriceColor(price, comparePrice):
	"""获取价格数据"""
	if gPaint.defaultUIStyle == "dark":
		if price != 0:
			if price > comparePrice:
				return "rgb(255,82,82)"
			elif price < comparePrice:
				return "rgb(46,255,50)"
		return "rgb(190,190,235)"
	else:
		if price != 0:
			if price > comparePrice:
				return "rgb(255,82,82)"
			elif price < comparePrice:
				return "rgb(0,200,0)"
		return "rgb(0,0,0)"

def findViewsByType(findType, views, refViews):
	"""查找同类型视图"""
	size = len(views)
	for i in range(0, size):
		view = views[i]
		if view.viewType == findType:
			refViews.append(view)
		elif len(view.views) > 0:
			findViewsByType(findType, view.views, refViews)

def onClickGridCell(grid, row, gridColumn, cell, firstTouch, firstPoint, secondTouch, secondPoint, clicks):
	"""点击单元格"""
	global currentCode
	code = row.cells[1].value
	name = row.cells[2].value
	queryHistoryData(code, name, 0, findMyCharts)
	queryHistoryData(code, name, 1, findMyCharts)
	queryHistoryData(code, name, 1440, findMyCharts)
	currentCode = code
	queryNewData()
	invalidate(grid.paint)

def onClick(view, firstTouch, firstPoint, secondTouch, secondPoint, clicks):
	"""视图的鼠标点击方法
	view 视图
	mp 坐标
	buttons 按钮 0未按下 1左键 2右键
	clicks 点击次数
	delta 滚轮值"""
	onClickDefault(view, firstTouch, firstPoint, secondTouch, secondPoint, clicks)
	if view.viewName.find("cycle,") == 0:
		strs = view.viewName.split(",")
		index = int(strs[1])
		cycleInt = int(strs[2])
		myChart = findMyCharts[index]
		myChart.exAttributes["cycle"] = str(cycleInt)
		queryCycle = 0
		if cycleInt > 0 and cycleInt < 1440:
			queryCycle = 1
		elif cycleInt >= 1440:
			queryCycle = 1440
		myCharts = []
		myCharts.append(myChart)
		chart = charts[index]
		code = chart.text.split(" ")[0]
		name = chart.text.split(" ")[1]
		if (chart.viewName == "preChart"):
			chart.datas2 = []
		queryHistoryData(code, name, queryCycle, myCharts)
	elif view.viewName == "preButton":
		if view.text == "预测中...":
			return
		chart = findViewByName("preChart", gPaint.views)
		thread = threading.Thread(target=predict_in_thread, args=(chart,))
		thread.start()
	elif view.viewType == "menuitem":
		name = view.viewName
		chart = findViewByName("preChart", gPaint.views)
		if name.startswith("mode"):
			chart.datas2 = []
			chart.preMode = name.split("_")[1]
		elif name.startswith("lookback"):
			chart.lookback = int(name.split("_")[1])
		elif name.startswith("pred_len"):
			chart.datas2 = []
			chart.pred_len = int(name.split("pred_len_")[1])
	elif view.viewName == "temperatureUp":
		chart = findViewByName("preChart", gPaint.views)
		temperatureDiv = findViewByName("temperatureDiv", gPaint.views)
		temp = chart.temperature
		if temp < 100:
			chart.temperature += 1.0
			temperatureDiv.text = "Temp:" + str(toFixed(chart.temperature, 1))
		elif temp < 1:
			chart.temperature += 0.1
			temperatureDiv.text = "Temp:" + str(toFixed(chart.temperature, 1))
		temperatureDiv.parent.invalidate()
	elif view.viewName == "temperatureDown":
		temperatureDiv = findViewByName("temperatureDiv", gPaint.views)
		chart = findViewByName("preChart", gPaint.views)
		temp = chart.temperature
		if temp > 3:
			chart.temperature -= 1.0
			temperatureDiv.text = "Temp:" + str(toFixed(chart.temperature, 1))
		elif temp > 0.1:
			chart.temperature -= 0.1
			temperatureDiv.text = "Temp:" + str(toFixed(chart.temperature, 1))
		temperatureDiv.parent.invalidate()
	elif view.viewName == "topPUp":
		topPDiv = findViewByName("topPDiv", gPaint.views)
		chart = findViewByName("preChart", gPaint.views)
		topP = chart.topP
		if topP < 1:
			chart.topP += 0.1
			topPDiv.text = "topP:" + str(toFixed(chart.topP, 1))
		topPDiv.parent.invalidate()
	elif view.viewName == "topPDown":
		topPDiv = findViewByName("topPDiv", gPaint.views)
		chart = findViewByName("preChart", gPaint.views)
		topP = chart.topP
		if topP > 0:
			chart.topP -= 0.1
			topPDiv.text = "topP:" + str(toFixed(chart.topP, 1))
		topPDiv.parent.invalidate()

def createGridCell (grid):
	"""创建单元格"""
	gridCell = FCGridCell()
	if grid.paint.defaultUIStyle == "dark":
		gridCell.backColor = "none"
		gridCell.borderColor = "none"
		gridCell.textColor = "rgb(175,196,228)"
	elif grid.paint.defaultUIStyle == "light":
		gridCell.backColor = "none"
		gridCell.borderColor = "none"
		gridCell.textColor = "rgb(0,0,0)"
	gridCell.font = "Default,13"
	return gridCell

def queryPriceCallBack(data):
	"""板块数据回调"""
	global priceRowMap
	if data.success:
		global gridStocks
		result = data.data
		strs = result.split("\r\n")
		for i in range(0, len(strs)):
			subStrs = strs[i].split(",")
			if len(subStrs) >= 15:
				row = None
				cell1 = None
				cell2 = None
				cell3 = None
				cell4 = None
				cell5 = None
				cell6 = None
				cell7 = None
				cell8 = None
				cell9 = None
				cell10 = None
				cell11 = None
				cell12 = None
				cell13 = None
				cell14 = None
				cell15 = None
				cell16 = None
				cell17 = None
				cell18 = None
				cell19 = None
				cell20 = None
				cell21 = None
				cell22 = None
				code = subStrs[0]
				if code in priceRowMap:
					row = priceRowMap[code]
					cell1 = row.cells[0]
					cell2 = row.cells[1]
					cell3 = row.cells[2]
					cell4 = row.cells[3]
					cell5 = row.cells[4]
					cell6 = row.cells[5]
					cell7 = row.cells[6]
					cell8 = row.cells[7]
					cell9 = row.cells[8]
					cell10 = row.cells[9]
					cell11 = row.cells[10]
					cell12 = row.cells[11]
					cell13 = row.cells[12]
					cell14 = row.cells[13]
					cell15 = row.cells[14]
					cell16 = row.cells[15]
					cell17 = row.cells[16]
					cell18 = row.cells[17]
					cell19 = row.cells[18]
					cell20 = row.cells[19]
					cell21 = row.cells[20]
					cell22 = row.cells[21]
				else:
					row = FCGridRow()
					priceRowMap[code] = row
					gridStocks.rows.append(row)
					cell1 = createGridCell(gridStocks)
					row.cells.append(cell1)
					cell2 = createGridCell(gridStocks)
					row.cells.append(cell2)
					cell3 = createGridCell(gridStocks)
					row.cells.append(cell3)
					cell4 = createGridCell(gridStocks)
					row.cells.append(cell4)
					cell5 = createGridCell(gridStocks)
					row.cells.append(cell5)
					cell6 = createGridCell(gridStocks)
					row.cells.append(cell6)
					cell7 = createGridCell(gridStocks)
					row.cells.append(cell7)
					cell8 = createGridCell(gridStocks)
					row.cells.append(cell8)
					cell9 = createGridCell(gridStocks)
					row.cells.append(cell9)
					cell10 = createGridCell(gridStocks)
					row.cells.append(cell10)
					cell11 = createGridCell(gridStocks)
					row.cells.append(cell11)
					cell12 = createGridCell(gridStocks)
					row.cells.append(cell12)
					cell13 = createGridCell(gridStocks)
					row.cells.append(cell13)
					cell14 = createGridCell(gridStocks)
					row.cells.append(cell14)
					cell15 = createGridCell(gridStocks)
					row.cells.append(cell15)
					cell16 = createGridCell(gridStocks)
					row.cells.append(cell16)
					cell17 = createGridCell(gridStocks)
					row.cells.append(cell17)
					cell18 = createGridCell(gridStocks)
					row.cells.append(cell18)
					cell19 = createGridCell(gridStocks)
					row.cells.append(cell19)
					cell20 = createGridCell(gridStocks)
					row.cells.append(cell20)
					cell21 = createGridCell(gridStocks)
					row.cells.append(cell21)
					cell22 = createGridCell(gridStocks)
					row.cells.append(cell22)
					cell1.value = len(gridStocks.rows)

					cell2.value = subStrs[0]
					cell2.textColor = "rgb(194,151,18)"
					
					cell3.value = subStrs[1]

				close = float(subStrs[2])
				high = float(subStrs[3])
				low =  float(subStrs[4])
				lastClose = float(subStrs[8])
				cell4.value = toFixed(close, 2)
				cell4.textColor = getPriceColor(close, lastClose)

				diff = 0
				if lastClose > 0:
					diff = 100 * (close - lastClose) / lastClose
				cell5.value = toFixed(diff, 2) + "%"
				cell5.textColor = getPriceColor(diff, 0)

				cell6.value = toFixed(close - lastClose, 2)
				cell6.textColor = getPriceColor(close, lastClose)

				volume = float(subStrs[6])
				amount = float(subStrs[7])
				cell7.value = toFixed(volume / 100 / 10000, 2) + "万"

				cell8.value = toFixed(amount / 100000000, 2) + "亿"

				cell9.value = toFixed(float(subStrs[12]), 2)

				cell10.value = toFixed(float(subStrs[11]), 2)

				diff2 = 0
				if lastClose > 0:
					diff2 = 100 * (high - lastClose) / lastClose - 100 * (low - lastClose) / lastClose
				cell11.value = toFixed(diff2, 2) + "%"

				cell12.value = toFixed(float(subStrs[13]), 2)

				marketValue = float(subStrs[9]) * close
				cell13.value = toFixed(marketValue / 100000000, 2) + "亿"

				flowValue = float(subStrs[10]) * close
				cell14.value = toFixed(flowValue / 100000000, 2) + "亿"

				cell15.value = ""

				upperLimit = float(subStrs[14])
				lowerLimit = float(subStrs[15])
				cell16.value = toFixed(upperLimit, 2)
				cell16.textColor = getPriceColor(1, 0)

				cell17.value = toFixed(lowerLimit, 2)
				cell17.textColor = getPriceColor(0, 1)

				cell18.value = ""

				cell19.value = ""

				cell20.value = ""

				cell21.value = ""

				cell22.value = ""
		gridStocks.invalidate()

def queryPrice():
	"""查询报价数据"""
	url = "http://www.jjmfc.com:9968/quote?func=price&codes=all&count=200"
	tag = []
	httpRequest(url, queryPriceCallBack, tag)

def drawChartHScale(chart, paint, clipRect):
	"""绘制横轴刻度的自定义方法
	chart:图表
	paint:绘图对象
	clipRect:裁剪区域"""
	#判断数据是否为空
	if chart.datas != None and len(chart.datas) > 0 and chart.hScaleHeight > 0:
		if chart.cycle == "trend":
			times = []
			if chart.size.cx < 600:
				times.append(10 * 60 + 30)
				times.append(11 * 60 + 30)
				times.append(14 * 60)
			else:
				times.append(10 * 60)
				times.append(10 * 60 + 30)
				times.append(11 * 60)
				times.append(11 * 60 + 30)
				times.append(13 * 60 + 30)
				times.append(14 * 60)
				times.append(14 * 60 + 30)
			for i in range(chart.firstVisibleIndex, chart.lastVisibleIndex + 1):
				dateNum = chart.datas[i].date
				date = time.localtime(dateNum)
				hour = date.tm_hour
				minute = date.tm_min
				for j in range(0, len(times)):
					if times[j] == hour * 60 + minute:
						x = getChartX(chart, i)
						bBottom = chart.size.cy
						paint.drawLine(chart.scaleColor, 1, 0, x, bBottom - chart.hScaleHeight, x, bBottom - chart.hScaleHeight + 12)
						paint.drawLine(chart.gridColor, 1, 0, x, 0, x, bBottom - chart.hScaleHeight)
						xText = time.strftime("%H:%M", date)
						tSize = paint.textSize(xText, "Default,12")
						paint.drawText(xText, chart.hScaleTextColor, "Default,12", x - tSize.cx / 2, bBottom - chart.hScaleHeight / 2 - tSize.cy / 2)
						break
		elif chart.cycle == "minute":
			lastYear = 0
			lastDate2 = 0
			dLeft = chart.leftVScaleWidth
			i = chart.firstVisibleIndex
			while i <= chart.lastVisibleIndex:
				dateNum = chart.datas[i].date
				date = time.localtime(dateNum)
				year = date.tm_year
				xText = ""
				if year != lastYear:
					xText = time.strftime("%Y/%m/%d", date)
				else:
					xText = time.strftime("%m/%d", date)
				lastDate = time.localtime(lastDate2)
				if int(date.tm_year * 10000 + date.tm_mon * 100 + date.tm_mday) != int(lastDate.tm_year * 10000 + lastDate.tm_mon * 100 + lastDate.tm_mday):
					lastDate2 = dateNum
					lastYear = year
					tSize = paint.textSize(xText, "Default,12")
					x = getChartX(chart, i)
					dx = x + 2
					if dx > dLeft and dx + tSize.cx < chart.size.cx - chart.rightVScaleWidth - 5:
						bBottom = chart.size.cy
						paint.drawLine(chart.scaleColor, 1, 0, x, bBottom - chart.hScaleHeight, x, bBottom - chart.hScaleHeight + 12)
						paint.drawText(xText, chart.hScaleTextColor, "Default,12", dx, bBottom - chart.hScaleHeight / 2 - tSize.cy / 2)
						i = i + int((tSize.cx + chart.hScaleTextDistance) / chart.hScalePixel) + 1
				i = i + 1						
		else:
			drawLeft = chart.leftVScaleWidth #左侧起画点
			i = chart.firstVisibleIndex #开始索引
			lastYear = 0 #缓存年份，用于判断是否换年
			drawYearsCache = [] #实际绘制到图形上的年份文字
			lastTextRight = 0 #上个文字的右侧
			timeCache = [] #保存日期的缓存
			yearTextLeftCache = [] #绘制年文字的左侧位置缓存
			yearTextRightCache = [] #绘制年文字的右侧位置缓存
			textPadding = 5 #两个文字之间的最小间隔
			#逐步递增索引，先绘制年
			while i <= chart.lastVisibleIndex:
				dateObj = time.localtime(chart.datas[i].date) #将时间戳转换为time，并缓存到集合中
				timeCache.append(dateObj)
				year = dateObj.tm_year #从结构中获取年份			
				x = getChartX(chart, i) #获取索引对应的位置
				#判断是否换年，以及是否在绘图区间内
				if year != lastYear and x >= drawLeft and x < chart.size.cx - chart.rightVScaleWidth:
					month = dateObj.tm_mon #获取月的结构
					xText = str(year) #拼接要绘制的文字
					if month < 10:
						xText = xText + "/0" + str(month) #如果小于10月要补0
					else:
						xText = xText + "/" + str(month) #大于等于10月不用补0
					tSize = paint.textSize(xText, chart.font) #计算要绘制文字的大小
					paint.drawLine(chart.scaleColor, 1, 0, x, chart.size.cy - chart.hScaleHeight, x, chart.size.cy - chart.hScaleHeight + 8) #绘制刻度线
					#判断是否和上个文字重影
					if x + 2 > lastTextRight + textPadding:
						paint.drawText(xText, chart.hScaleTextColor, "Default,12", x + 2, chart.size.cy - chart.hScaleHeight + 8  - tSize.cy / 2 + 7) #绘制文字
						yearTextLeftCache.append(x + 2) #将年文字的左侧位置缓存
						yearTextRightCache.append(x + 2 + tSize.cx) #将年文字的右侧位置缓存
						drawYearsCache.append(year) #缓存要绘制的年
						lastTextRight = x + 2 + tSize.cx #缓存上个文字的右侧位置
					lastYear = year #记录上次绘制的年份
				i = i + 1	#索引累加	
			#绘制月份
			for m in range(0, len(drawYearsCache)):
				cacheYear = drawYearsCache[m] #从缓存中获取年份
				lastMonth = 0 #缓存月份，用于判断是否换月
				i = chart.firstVisibleIndex #重置开始索引
				lastTextRight = 0 #重置上个文字的右侧
				#逐步递增索引
				while i <= chart.lastVisibleIndex:
					dateObj = timeCache[i - chart.firstVisibleIndex] #从缓存中获取time
					year = dateObj.tm_year #从结构中获取年份
					#判断是否同一年	
					if cacheYear == year:
						month = dateObj.tm_mon #从结构中获取月份
						x = getChartX(chart, i)
						#判断是否换月，以及是否在绘图区间内
						if lastMonth != month and x >= drawLeft and x < chart.size.cx - chart.rightVScaleWidth:			
							xText = str(month) #获取绘制的月份文字
							tSize = paint.textSize(xText, chart.font) #计算要绘制文字的大小
							#判断是否和上个文字重影
							if x + 2 > lastTextRight + textPadding:
								#判断是否和年的文字重影
								if (x + 2 > yearTextRightCache[m] + textPadding) and ((m == len(drawYearsCache) - 1) or (m < len(drawYearsCache) - 1 and x + 2 + tSize.cx < yearTextLeftCache[m + 1] - textPadding)):
									paint.drawLine(chart.scaleColor, 1, 0, x, chart.size.cy - chart.hScaleHeight, x, chart.size.cy - chart.hScaleHeight + 6) #绘制刻度
									paint.drawText(xText, chart.hScaleTextColor, "Default,12", x + 2, chart.size.cy - chart.hScaleHeight + 8  - tSize.cy / 2 + 7) #绘制文字
									lastTextRight = x + 2 + tSize.cx #缓存上个文字的右侧位置
							lastMonth = month #记录上次绘制的月份
					elif cacheYear < year:
						break #超过区间，退出循环
					i = i + 1	#索引累加

def drawChartTip(chart, paint, clipRect):
	"""绘制图表提示
	chart:图表
	paint:绘图对象
	clipRect:裁剪区域"""
	if paint.touchMoveView == chart and chart.cycle != "trend":
		crossLineIndex = chart.crossStopIndex
		if crossLineIndex != -1 and crossLineIndex >= chart.firstVisibleIndex and crossLineIndex <= chart.lastVisibleIndex:
			mp = chart.touchPosition
			cmp = FCPoint(mp.x + 5, mp.y)
			width = 125
			height = 125
			tipRect = FCRect(cmp.x, cmp.y, cmp.x + width, cmp.y + height)
			if tipRect.left < 0:
				tipRect.left = 0
				tipRect.right = width
			if tipRect.right > chart.size.cx:
				tipRect.left = chart.size.cx - width
				tipRect.right = tipRect.left + width
			if tipRect.bottom > chart.size.cy:
				tipRect.top = chart.size.cy - height
				tipRect.bottom = tipRect.top + height
			sData = chart.datas[crossLineIndex]
			high = sData.high
			low = sData.low
			highY = getChartY(chart, 0, high)
			lowY = getChartY(chart, 0, low)
			if cmp.y >= highY and cmp.y <= lowY:
				if paint.defaultUIStyle == "dark":
					paint.fillRect("rgb(50,50,50)", tipRect.left, tipRect.top, tipRect.right, tipRect.bottom)
				elif paint.defaultUIStyle == "light":
					paint.fillRect("rgb(220,220,220)", tipRect.left, tipRect.top, tipRect.right, tipRect.bottom)
				close = sData.close
				openPrice = sData.open
				volume = sData.volume
				lastClose = sData.open
				if crossLineIndex > 0:
					lastClose = chart.datas[crossLineIndex - 1].close
				xText = ""
				if chart.cycle == "day":
					timeArray = time.localtime(sData.date)
					xText = time.strftime("%Y-%m-%d", timeArray)
				elif chart.cycle == "minute":
					timeArray = time.localtime(sData.date)
					xText = time.strftime("%Y-%m-%d %H:%M", timeArray)
				xFont = "Default,13"
				paint.drawText(xText, chart.textColor, xFont, tipRect.left + 5, tipRect.top + 5)
				paint.drawText("高:", chart.textColor, xFont, tipRect.left + 5, tipRect.top + 25)
				paint.drawText(toFixed(high, chart.candleDigit), getPriceColor(high, lastClose), xFont, tipRect.left + 25, tipRect.top + 25)
				paint.drawText("开:", chart.textColor, xFont, tipRect.left + 5, tipRect.top + 45)
				paint.drawText(toFixed(openPrice, chart.candleDigit), getPriceColor(openPrice, lastClose), xFont, tipRect.left + 25, tipRect.top + 45)
				paint.drawText("低:", chart.textColor, xFont, tipRect.left + 5, tipRect.top + 65)
				paint.drawText(toFixed(low, chart.candleDigit), getPriceColor(low, lastClose), xFont, tipRect.left + 25, tipRect.top + 65)
				paint.drawText("收:", chart.textColor, xFont, tipRect.left + 5, tipRect.top + 85)
				paint.drawText(toFixed(close, chart.candleDigit), getPriceColor(close, lastClose), xFont, tipRect.left + 25, tipRect.top + 85)
				paint.drawText("量:", chart.textColor, xFont, tipRect.left + 5, tipRect.top + 105)
				paint.drawText(toFixed(volume, 0), "rgb(80,255,255)", xFont, tipRect.left + 25, tipRect.top + 105)

def WndProcPopup(hwnd,msg,wParam,lParam):
	"""消息循环"""
	global popupPaint
	if (hwnd in popupPaint) == True:
		return WndProcDefault(popupPaint[hwnd],hwnd,msg,wParam,lParam)
	else:
		return DefWindowProc(hwnd,msg,wParam,lParam)

popupPaint = dict()
# 子窗体
def popupWindow(text):
	global popupPaint
	pPaint = FCPaint() #创建绘图对象
	wSize = FCSize(500, 200)
	setWindowSize(wSize)
	setCenterScreen(True)
	#初始化窗体
	createWindow(pPaint, "提示", WndProcPopup)
	popupPaint[pPaint.hWnd] = pPaint
	# 转义文本，避免XML注入/渲染失败
	safe_text = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
	xml = f"""<?xml version="1.0" encoding="UTF-8"?>
		<html xmlns="facecat">
		<body>
			<div dock="fill" backcolor="rgb(170,178,189)">
				<label name="labelText" size="350,33" text="{safe_text}" font="Default,30" location="100,20"/>
				<label name="labelText2" size="350,33" text="请切换周期或修改训练参数。" font="Default,30" location="100,55"/>
			</div>
		</body>
		</html>"""
	renderFaceCat(pPaint, xml)
	showWindow(pPaint)
	# 模态阻塞：直到窗口被关闭
	msg = cwintypes.MSG()
	pmsg = ct.byref(msg)
	while user32.IsWindow(pPaint.hWnd):
		res = user32.GetMessageW(pmsg, None, 0, 0)
		if res <= 0:
			break
		user32.TranslateMessage(pmsg)
		user32.DispatchMessageW(pmsg)
def WndProc(hwnd,msg,wParam,lParam):
	"""消息循环"""
	if msg == 0x0401:
		gPaint.dealData()
	return WndProcDefault(gPaint,hwnd,msg,wParam,lParam)

gPaint = FCPaint() #创建绘图对象
gPaint.defaultUIStyle = "dark"
gPaint.onPaint = onPaint
gPaint.onClickGridCell = onClickGridCell
gPaint.onClick = onClick
gPaint.onPaintChartHScale = drawChartHScale

gPaint.highQuanlity()
gPaint.scaleFactorX = 1.33
gPaint.scaleFactorY = 1.33

#初始化窗体
createMainWindow(gPaint, "Facecat-Kronos", WndProc)
xml = """<?xml version="1.0" encoding="utf-8" ?>
	    <html xmlns="facecat">
	    <body>
	        <div bordercolor="none" name="divInner" dock="fill">
	            <div type="tab" dock="fill" selectedindex="0" backcolor="none" bordercolor="none"
	            name="tabFunc">
	                <div type="tabpage" text="主界面" name="divMain" backcolor="none">
	                <div type="splitlayout" layoutstyle="lefttoright" bordercolor="none" dock="fill"
	                        size="650,510" candragsplitter="true" splitterposition="650,1" name="divCodingRight2">
	                        <table name="gridStocks" headerheight="30" dock="fill"
	                            gridlinecolor="none" bordercolor="none" showvscrollbar="true" showhscrollbar="true"
	                            allowpreviewsevent="true" allowdragscroll="true">
	                            <tr>
	                                <th name="colP0" text="序" width="40" allowdrag="true" allowresize="true" coltype="no"/>
	                                <th name="colP1" text="代码" width="70" allowdrag="true" allowresize="true"/>
	                                <th name="colP2" text="名称" width="70" allowdrag="true" allowresize="true" />
	                                <th name="colP3" text="现价" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP4" text="涨幅" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP5" text="涨跌" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP9" text="总量" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP10" text="总额" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP11" text="量比" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP12" text="PE动" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP13" text="振幅" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP14" text="换手率" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP15" text="总市值" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP16" text="流值" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP17" text="行业分类板块" width="80" allowdrag="true" allowresize="true" cellalign="center"/>
	                                <th name="colP18" text="涨停价" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP19" text="跌停价" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP20" text="金比" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP21" text="涨跌比" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP22" text="涨速" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP23" text="净资产收益率" width="100" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP24" text="自设指标" width="80" allowdrag="true" allowresize="true" cellalign="right"/>
	                            </tr>
	                        </table>
							<div type="tab" dock="fill" selectedindex="0" backcolor="none" bordercolor="none"
	           					 name="tabFunc">
								<div type="tabpage" text="预测" name="preTab" backcolor="none">
									<div type="splitlayout" layoutstyle="toptobottom" bordercolor="none" dock="fill" size="400,400" candragsplitter="true" splitterposition="25,1" >
										<div type="splitlayout" layoutstyle="righttoleft" bordercolor="none" dock="fill" size="400,400" candragsplitter="true" splitterposition="300,1" splitmode="percentsize">
											<div type="layout" backcolor="none" allowResize="false" bordercolor="none">
												<input type="button" name="preButton" text="预测" textcolor="rgb(255,255,255)" /> 
												<select name="modeSwich" selectedindex="0">
													<option text="预测模式" name="mode_predict" value="predict"/>
													<option text="回测模式" name="mode_backtest" value="backtest"/>
												</select>
											</div>
											<div type="layout" backcolor="none" bordercolor="none">
												<select name="lookback" selectedindex="0" size="80,25">
													<option text="训练 50" name="lookback_50" value="50"/>
													<option text="训练 10" name="lookback_10" value="10"/>
													<option text="训练 20" name="lookback_20" value="20"/>
													<option text="训练 100" name="lookback_100" value="100"/>
													<option text="训练 200" name="lookback_200" value="200"/>
													<option text="训练 300" name="lookback_300" value="300"/>
													<option text="训练 400" name="lookback_400" value="400"/>
												</select>
												<select name="pred_len" size="80,25">
													<option text="预测 5" name="pred_len_5" value="5"/>
													<option text="预测 1" name="pred_len_1" value="1"/>
													<option text="预测 10" name="pred_len_10" value="10"/>
													<option text="预测 20" name="pred_len_20" value="20"/>
													<option text="预测 50" name="pred_len_50" value="50"/>
													<option text="预测 120" name="pred_len_120" value="120"/>
												</select>
												<div size="80,25" name="temperatureDiv" text="Temp:1.0">
													<input type="button" location="65,4" size="10,12" name="temperatureUp" hoveredColor="rgb(180,180,180)" pushedColor="rgb(100,100,100)"/>
													<input type="button" location="65,8" size="10,12" name="temperatureDown" hoveredColor="rgb(180,180,180)" pushedColor="rgb(100,100,100)"/>
												</div>
												<div size="80,25" name="topPDiv" text="topP:0.9">
													<input type="button" location="65,4" size="10,12" name="topPUp" hoveredColor="rgb(180,180,180)" pushedColor="rgb(100,100,100)"/>
													<input type="button" location="65,8" size="10,12" name="topPDown" hoveredColor="rgb(180,180,180)" pushedColor="rgb(100,100,100)"/>
												</div>
												<div name="progress" bordercolor="none" text="0" />
											</div>
											
										</div>
										<div type="custom" cid="mychart" name="preDiv" cycle="1440" size="200,200" nativerefresh="true" 
										     candledivpercent="0.7"  voldivpercent="0.3"  />
									</div>
								</div>
								<div type="tabpage" text="主界面" name="divMain" backcolor="none">
									<div type="splitlayout" layoutstyle="toptobottom" bordercolor="none" dock="fill"
										size="400,400" candragsplitter="true" splitterposition="380,1">
										<div type="splitlayout" layoutstyle="righttoleft" bordercolor="none" dock="fill"
											size="400,400" candragsplitter="true" splitterposition="200,1">
											<div type="custom" cid="latestdiv" name="divLatest"/>
											<div type="custom" cid="mychart" name="mainChart1" cycle="0" candledivpercent="0.7"
													voldivpercent="0.3" backcolor="none" bordercolor="none"/>
										</div>
										<div type="splitlayout" layoutstyle="lefttoright" bordercolor="none" dock="fill"
											size="400,400" candragsplitter="true" splitmode="percentsize" splitterposition="200,1">
											<div type="custom" cid="mychart" name="mainChart2" cycle="5" candledivpercent="0.7"
													voldivpercent="0.3" backcolor="none" bordercolor="none"/>
											<div type="custom" cid="mychart" name="mainChart3" cycle="1440" candledivpercent="0.7"
													voldivpercent="0.3" backcolor="none" bordercolor="none"/>
										</div>
									</div>
								</div>
							</div>

	                    </div>
	                    </div>
	                    <div type="tabpage" text="多K线" name="divMulti" backcolor="none">
	                        <div type="splitlayout" layoutstyle="lefttoright" backcolor="none" bordercolor="none"
	                            dock="fill" size="400,400" candragsplitter="true" splitmode="percentsize" splitterposition="133,1">
	                            <div type="splitlayout" layoutstyle="toptobottom" backcolor="none" bordercolor="none"
	                                dock="fill" size="400,400" candragsplitter="true" splitmode="percentsize" splitterposition="267,1">
	                                <div type="splitlayout" layoutstyle="toptobottom" backcolor="none" bordercolor="none"
	                                    dock="fill" size="400,400" candragsplitter="true" splitmode="percentsize" splitterposition="200,1">
	                                    <div type="custom" cid="mychart" name="chart1" cycle="1" nativerefresh="true" candledivpercent="1"
	                                        voldivpercent="0" backcolor="none" bordercolor="none"/>
	                                    <div type="custom" cid="mychart" name="chart2" cycle="5" nativerefresh="true" candledivpercent="1"
	                                        voldivpercent="0" backcolor="none" bordercolor="none"/>
	                                </div>
	                                <div type="custom" cid="mychart" name="chart3" cycle="10" nativerefresh="true" candledivpercent="1"
	                                    voldivpercent="0" backcolor="none" bordercolor="none"/>
	                            </div>
	                            <div type="splitlayout" layoutstyle="lefttoright" backcolor="none" bordercolor="none"
	                                dock="fill" size="400,400" candragsplitter="true" splitmode="percentsize" splitterposition="200,1">
	                                <div type="splitlayout" layoutstyle="toptobottom" backcolor="none" bordercolor="none"
	                                    dock="fill" size="400,400" candragsplitter="true" splitmode="percentsize" splitterposition="267,1">
	                                    <div type="splitlayout" layoutstyle="toptobottom" backcolor="none" bordercolor="none"
	                                        dock="fill" size="400,400" candragsplitter="true" splitmode="percentsize" splitterposition="200,1">
	                                        <div type="custom" cid="mychart" name="chart4" cycle="15" nativerefresh="true" candledivpercent="1"
	                                            voldivpercent="0" backcolor="none" bordercolor="none"/>
	                                        <div type="custom" cid="mychart" name="chart5" cycle="20" nativerefresh="true" candledivpercent="1"
	                                            voldivpercent="0" backcolor="none" bordercolor="none"/>
	                                    </div>
	                                    <div type="custom" cid="mychart" name="chart6" cycle="30" nativerefresh="true" candledivpercent="1"
	                                        voldivpercent="0" backcolor="none" bordercolor="none"/>
	                                </div>
	                                <div type="splitlayout" layoutstyle="toptobottom" backcolor="none" bordercolor="none"
	                                    dock="fill" size="400,400" candragsplitter="true" splitmode="percentsize" splitterposition="267,1">
	                                    <div type="splitlayout" layoutstyle="toptobottom" backcolor="none" bordercolor="none"
	                                        dock="fill" size="400,400" candragsplitter="true" splitmode="percentsize" splitterposition="200,1">
	                                        <div type="custom" cid="mychart" name="chart7" cycle="1440" nativerefresh="true"
	                                            candledivpercent="1" voldivpercent="0" backcolor="none" bordercolor="none"/>
	                                        <div type="custom" cid="mychart" name="chart8" cycle="10080" nativerefresh="true"
	                                            candledivpercent="1" voldivpercent="0" backcolor="none" bordercolor="none"/>
	                                    </div>
	                                    <div type="custom" cid="mychart" name="chart9" cycle="43200" nativerefresh="true"
	                                        candledivpercent="1" voldivpercent="0" backcolor="none" bordercolor="none"/>
	                                </div>
	                            </div>
	                        </div>
	                    </div>
	                </div>
	            </div>
	    </body>
	    </html>"""

gPaint.render(None, xml)
gridStocks = gPaint.findView("gridStocks")
for i in range(3, len(gridStocks.columns)):
	gridStocks.columns[i].cellAlign = "right"
if gPaint.defaultUIStyle == "dark":
	gridStocks.selectedRowColor = "rgb(75,75,75)"
	gridStocks.alternateRowColor = "rgb(25,25,25)"
elif gPaint.defaultUIStyle == "light":
	gridStocks.selectedRowColor = "rgb(175,175,175)"
	gridStocks.alternateRowColor = "rgb(245,245,245)"
stockName = "浦发银行"

progressDiv = findViewByName("progress", gPaint.views)
progressDiv.onPaint = drawProgressDiv

temperatureUpButton = findViewByName("temperatureUp", gPaint.views)
temperatureUpButton.onPrePaint = drawUpButton
temperatureDownButton = findViewByName("temperatureDown", gPaint.views)
temperatureDownButton.onPrePaint = drawDownButton

topPUpButton = findViewByName("topPUp", gPaint.views)
topPUpButton.onPrePaint = drawUpButton
topPDownButton = findViewByName("topPDown", gPaint.views)
topPDownButton.onPrePaint = drawDownButton

temperatureDiv = findViewByName("temperatureDiv", gPaint.views)
temperatureDiv.onPaint = drawMyDiv
topPDiv = findViewByName("topPDiv", gPaint.views)
topPDiv.onPaint = drawMyDiv
topPDiv.parent.invalidate()
findViewsByType("mychart", gPaint.views, findMyCharts)
for i in range(0, len(findMyCharts)):
	myChart = findMyCharts[i]
	splitDiv = FCSplitLayoutDiv()
	splitDiv.layoutStyle = "toptobottom"
	splitDiv.size = FCSize(400, 400)
	splitDiv.backColor = "none"
	splitDiv.borderColor = "none"
	splitDiv.dock = "fill"
	myChart.addView(splitDiv)

	topDiv = FCLayoutDiv()
	topDiv.backColor = "none"
	topDiv.borderColor = "none"
	topDiv.layoutStyle = "lefttoright"
	topDiv.showHScrollBar = False

	bottomDiv = FCDiv()
	bottomDiv.backColor = "none"
	bottomDiv.borderColor = "none"

	addViewToSplit(splitDiv, topDiv, bottomDiv, 30)
	if i ==1:
		splitDiv.splitter.location = FCPoint(0, 0)
		
	chart = FCChart()
	chart.leftVScaleWidth = 70
	chart.rightVScaleWidth = 70
	if i > 0:
		chart.leftVScaleWidth = 0
	chart.vScaleDistance = 35
	chart.hScalePixel = 11
	chart.hScaleHeight = 30
	chart.candlePaddingTop = 30
	chart.candlePaddingBottom = 20
	chart.volPaddingTop = 20
	chart.volPaddingBottom = 0
	chart.vScaleDistance = 35
	chart.dock = "fill"
	chart.font = "Default,12"
	chart.candleDivPercent = float(myChart.exAttributes["candledivpercent"])
	chart.volDivPercent = float(myChart.exAttributes["voldivpercent"])
	chart.indDivPercent = 0
	chart.text = currentCode + " " + stockName
	chart.allowDragChartDiv = True
	chart.onPaintChartTip = drawChartTip
	
	charts.append(chart)
	bottomDiv.addView(chart)
	setChartTheme(chart, i)
	# 预测面板属性调整
	if myChart.viewName=="preDiv":
		chart.viewName = "preChart"
		chart.indicatorColors = []
		chart.indicatorColors.append("rgb(255,255,255)")
		chart.indicatorColors.append("rgb(255,255,0)")
		chart.indicatorColors.append("rgb(150,0,150)")
		chart.indicatorColors.append("rgb(255,0,0)")
		chart.indicatorColors.append("rgb(0,150,150)")
		chart.indicatorColors.append("rgb(0,150,0)")
		chart.indicatorColors.append("rgb(59,174,218)")
		chart.indicatorColors.append("rgb(50,50,50)")
		chart.onPaintChartStock = drawPreChart
		chart.leftVScaleWidth = 0
		chart.mainIndicator = "BOLL"
		chart.showIndicator = "BIAS"
		chart.preMode = "predict"
		chart.lookback = 50
		chart.pred_len = 5
		chart.datas2 = []
		chart.temperature = 1.0
		chart.topP = 0.9
		chart.upColor = "rgb(186,56,18)"
		chart.downColor = "rgb(31,182,177)"
		chart.upColor2 =  "rgba(188,56,18,100)"
		chart.downColor2 ="rgba(31,182,177,100)" 
		chart.vScaleTextColor = "rgb(255,255,255)"
	cycles = []
	cycles.append("1")
	cycles.append("5")
	cycles.append("10")
	cycles.append("15")
	cycles.append("20")
	cycles.append("30")
	cycles.append("60")
	cycles.append("90")
	cycles.append("120")
	cycles.append("日")
	cycles.append("周")
	cycles.append("月")
	cycles.append("季")
	cycles.append("半")
	cycles.append("年")
	cyclesInts = []
	cyclesInts.append(1)
	cyclesInts.append(5)
	cyclesInts.append(10)
	cyclesInts.append(15)
	cyclesInts.append(20)
	cyclesInts.append(30)
	cyclesInts.append(60)
	cyclesInts.append(90)
	cyclesInts.append(120)
	cyclesInts.append(1440)
	cyclesInts.append(10080)
	cyclesInts.append(43200)
	cyclesInts.append(129600)
	cyclesInts.append(259200)
	cyclesInts.append(518400)
	for c in range(0, len(cycles)):
		cycleButton = FCButton()
		cycleButton.text = cycles[c]
		cycleButton.size = FCSize(27, 30)
		if gPaint.defaultUIStyle == "dark":
			cycleButton.textColor = "rgb(200,200,200)"
			cycleButton.borderColor = "rgb(50,50,50)"
		elif gPaint.defaultUIStyle == "light":
			cycleButton.textColor = "rgb(50,50,50)"
			cycleButton.borderColor = "rgb(200,200,200)"
		cycleButton.backColor = "none"
		cycleButton.viewName = "cycle," + str(i) + "," + str(cyclesInts[c])
		topDiv.addView(cycleButton)
queryHistoryData(currentCode, stockName, 0, findMyCharts)
queryHistoryData(currentCode, stockName, 1, findMyCharts)
queryHistoryData(currentCode, stockName, 1440, findMyCharts)
queryPrice()
queryNewData()
thread = threading.Thread(target=startQueryNewData, args=())
thread.start()
thread2 = threading.Thread(target=startQueryPriceData, args=())
thread2.start()
gPaint.update()
showWindow(gPaint)
