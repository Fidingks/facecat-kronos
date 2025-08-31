from facecat import *
from datetime import datetime

class ClientTickDataCache:
	def __init__(self):
	    self.code = ""  # 初始化代码
	    self.lastAmount = 0  # 初始化上次成交额
	    self.lastDate = 0  # 初始化上次日期
	    self.lastVolume = 0  # 初始化上次成交量

class ADJUSTMENTFACTOR:
	def __init__(self):
	    self.dwDate = 0  # 初始化日期
	    self.f1 = 0  # 每10股派现
	    self.f2 = 0  # 配股价
	    self.f3 = 0  # 每10股送股
	    self.f4 = 0  # 每10股配股

def getDateNum(year, month, day, hour, minute, second, millisecond):
	""" 获取日期的时间戳
	 year 年份
	 month 月份
	 day 日
	 hour 小时
	 minute 分钟
	 second 秒
	 millisecond 毫秒
	 @returns 返回日期的时间戳"""
	date = datetime(year, month, day, hour, minute, second, millisecond)
	return int(date.timestamp())

def numToDate(num):
	""" 时间戳转日期
	 num 时间戳
	 @returns 返回日期对象"""
	date = datetime.fromtimestamp(num)
	return date

def getSeason(month):
	""" 获取季度
	 month 月份
	 @returns 返回季度"""
	if 1 <= month <= 3:
	    return 1
	elif 4 <= month <= 6:
	    return 2
	elif 7 <= month <= 9:
	    return 3
	else:
	    return 4

def copySecurityData(data):
	""" 拷贝数据
	 data 原来的数据
	 @returns 新数据"""
	newData = SecurityData()
	newData.date = data.date
	newData.high = data.high
	newData.low = data.low
	newData.open = data.open
	newData.close = data.close
	newData.amount = data.amount
	newData.volume = data.volume
	return newData

def multiMinuteSecurityDatas(newDatas, minuteDatas, cycle):
	""" 多分钟数据处理
	 newDatas 新数据数组
	 minuteDatas 分钟数据数组
	 cycle 周期"""
	lastMinutes = 0
	for minuteData in minuteDatas:
	    minutes = minuteData.date // 60
	    if lastMinutes == 0:
	        lastMinutes = minutes
	    # 更新
	    if newDatas and minutes - lastMinutes < cycle:
	        lastData = newDatas[len(newDatas) - 1]
	        lastData.close = minuteData.close
	        if minuteData.high > lastData.high:
	            lastData.high = minuteData.high
	        if minuteData.low < lastData.low:
	            lastData.low = minuteData.low
	        lastData.amount += minuteData.amount
	        lastData.volume += minuteData.volume
	    else:
	        newData = copySecurityData(minuteData)
	        newDatas.append(newData)
	        lastMinutes = minutes

def getHistoryWeekDatas(weekDatas, dayDatas):
	""" 获取历史周数据
	 weekDatas 周数据数组
	 dayDatas 日数据数组
	 @returns 返回操作结果"""
	dayDatasSize = len(dayDatas)
	if dayDatasSize > 0:
	    firstDate = getDateNum(1970, 1, 5, 0, 0, 0, 0)
	    weekData = copySecurityData(dayDatas[0])
	    lWeeks = (weekData.date - firstDate) // 86400 // 7
	    for i in range(dayDatasSize):
	        dayData = copySecurityData(dayDatas[i])
	        weeks = (dayData.date - firstDate) // 86400 // 7
	        isNextWeek = weeks > lWeeks
	        if isNextWeek:
	            weekDatas.append(weekData)
	            weekData = copySecurityData(dayData)
	            if i == dayDatasSize - 1:
	                weekDatas.append(weekData)
	        else:
	            if i > 0:
	                weekData.close = dayData.close
	                weekData.amount += dayData.amount
	                weekData.volume += dayData.volume
	                if weekData.high < dayData.high:
	                    weekData.high = dayData.high
	                if weekData.low > dayData.low:
	                    weekData.low = dayData.low
	            if i == dayDatasSize - 1:
	                weekDatas.append(weekData)
	        lWeeks = weeks
	return 1

def getHistoryMonthDatas(monthDatas, dayDatas):
	""" 获取历史月数据
	 monthDatas 月数据数组
	 dayDatas 日数据数组
	 返回操作结果"""
	dayDatasSize = len(dayDatas)
	if dayDatasSize > 0:
	    monthData = copySecurityData(dayDatas[0])
	    ldate = numToDate(monthData.date)
	    lYear = ldate.year
	    lMonth = ldate.month
	    lDay = ldate.day
	    for i in range(dayDatasSize):
	        dayData = copySecurityData(dayDatas[i])
	        date = numToDate(dayData.date)
	        year = date.year
	        month = date.month
	        day = date.day
	        isNextMonth = year * 12 + month > lYear * 12 + lMonth
	        if isNextMonth:
	            monthDatas.append(monthData)
	            monthData = copySecurityData(dayData)
	            if i == dayDatasSize - 1:
	                monthDatas.append(monthData)
	        else:
	            if i > 0:
	                monthData.close = dayData.close
	                monthData.amount += dayData.amount
	                monthData.volume += dayData.volume
	                if monthData.high < dayData.high:
	                    monthData.high = dayData.high
	                if monthData.low > dayData.low:
	                    monthData.low = dayData.low
	            if i == dayDatasSize - 1:
	                monthDatas.append(monthData)
	        lYear = year
	        lMonth = month
	        lDay = day
	return 1

def getHistorySeasonDatas(seasonDatas, dayDatas):
	""" 获取历史季节数据
	 seasonDatas 季节数据数组
	 dayDatas 日数据数组
	 @returns 返回操作结果"""
	dayDatasSize = len(dayDatas)
	if dayDatasSize > 0:
	    seasonData = copySecurityData(dayDatas[0])
	    ldate = numToDate(seasonData.date)
	    lYear = ldate.year
	    lMonth = ldate.month
	    lDay = ldate.day
	    for i in range(dayDatasSize):
	        dayData = copySecurityData(dayDatas[i])
	        date = numToDate(dayData.date)
	        year = date.year
	        month = date.month
	        day = date.day
	        isNextSeason = year * 4 + getSeason(month) > lYear * 4 + getSeason(lMonth)
	        if isNextSeason:
	            seasonDatas.append(seasonData)
	            seasonData = copySecurityData(dayData)
	            if i == dayDatasSize - 1:
	                seasonDatas.append(seasonData)
	        else:
	            if i > 0:
	                seasonData.close = dayData.close
	                seasonData.amount += dayData.amount
	                seasonData.volume += dayData.volume
	                if seasonData.high < dayData.high:
	                    seasonData.high = dayData.high
	                if seasonData.low > dayData.low:
	                    seasonData.low = dayData.low
	            if i == dayDatasSize - 1:
	                seasonDatas.append(seasonData)
	        lYear = year
	        lMonth = month
	        lDay = day
	return 1

def getHistoryHalfYearDatas(halfYearDatas, dayDatas):
	""" 获取历史半年数据
	 halfYearDatas 半年数据数组
	 dayDatas 日数据数组
	 @returns  返回操作结果"""
	dayDatasSize = len(dayDatas)
	if dayDatasSize > 0:
	    yearData = copySecurityData(dayDatas[0])
	    ldate = numToDate(yearData.date)
	    lyear = ldate.year
	    lmonth = ldate.month
	    for i in range(dayDatasSize):
	        dayData = copySecurityData(dayDatas[i])
	        date = numToDate(dayData.date)
	        year = date.year
	        month = date.month
	        isNextHalfYear = year * 2 + month // 6 > lyear * 2 + lmonth // 6
	        if isNextHalfYear:
	            halfYearDatas.append(yearData)
	            yearData = copySecurityData(dayData)
	            if i == dayDatasSize - 1:
	                halfYearDatas.append(yearData)
	        else:
	            if i > 0:
	                yearData.close = dayData.close
	                yearData.amount += dayData.amount
	                yearData.volume += dayData.volume
	                if yearData.high < dayData.high:
	                    yearData.high = dayData.high
	                if yearData.low > dayData.low:
	                    yearData.low = dayData.low
	            if i == dayDatasSize - 1:
	                halfYearDatas.append(yearData)
	        lyear = year
	        lmonth = month
	return 1

def getHistoryYearDatas(yearDatas, dayDatas):
	""" 获取历史年数据
	 yearDatas 年数据数组
	 dayDatas 日数据数组
	 @returns 返回操作结果"""
	dayDatasSize = len(dayDatas)
	if dayDatasSize > 0:
	    yearData = copySecurityData(dayDatas[0])
	    ldate = numToDate(yearData.date)
	    lyear = ldate.year
	    lmonth = ldate.month
	    for i in range(dayDatasSize):
	        dayData = copySecurityData(dayDatas[i])
	        date = numToDate(dayData.date)
	        year = date.year
	        month = date.month
	        isNextYear = year > lyear
	        if isNextYear:
	            yearDatas.append(yearData)
	            yearData = copySecurityData(dayData)
	            if i == dayDatasSize - 1:
	                yearDatas.append(yearData)
	        else:
	            if i > 0:
	                yearData.close = dayData.close
	                yearData.amount += dayData.amount
	                yearData.volume += dayData.volume
	                if yearData.high < dayData.high:
	                    yearData.high = dayData.high
	                if yearData.low > dayData.low:
	                    yearData.low = dayData.low
	            if i == dayDatasSize - 1:
	                yearDatas.append(yearData)
	        lyear = year
	        lmonth = month
	return 1

def mergeLatestData(code, oldDatas, latestData, tickDataCache, dCycle):
	""" 合并最新数据
	 code 代码
	 oldDatas 老数据数组
	 latestData 新数据对象
	 tickDataCache TICK数据缓存对象
	 dCycle 周期"""
	cycle = dCycle
	if cycle == 0:
	    cycle = 1
	if latestData.open <= 0 or latestData.volume <= 0 or latestData.amount <= 0:
	    return
	newDate = numToDate(latestData.date)
	hourMinute = newDate.hour * 60 + newDate.minute
	if hourMinute < 570:
	    newDate = newDate.replace(hour=9, minute=30, second=0, microsecond=0)
	    latestData.date = int(newDate.timestamp())
	elif hourMinute < 571:
	    newDate = newDate.replace(hour=9, minute=31, second=0, microsecond=0)
	    latestData.date = int(newDate.timestamp())
	elif hourMinute > 900:
	    newDate = newDate.replace(hour=15, minute=0, second=0, microsecond=0)
	    latestData.date = int(newDate.timestamp())
	elif hourMinute > 690 and hourMinute < 780:
	    newDate = newDate.replace(hour=11, minute=30, second=0, microsecond=0)
	    latestData.date = int(newDate.timestamp())
    
	isNextCycle = True
	if dCycle == 0:
	    isNextCycle = False
	elif cycle < 1440:
	    if len(oldDatas) > 0:
	        newMinutes = latestData.date // 60
	        lastData = oldDatas[len(oldDatas) - 1]
	        lastMinutes = lastData.date // 60
	        isNextCycle = newMinutes - lastMinutes >= cycle
	else:
	    if cycle == 1440:
	        if len(oldDatas) > 0:
	            lastDate = numToDate(oldDatas[len(oldDatas) - 1].date)
	            isNextCycle = getDateNum(newDate.year, newDate.month, newDate.day, 0, 0, 0, 0) != getDateNum(lastDate.year, lastDate.month, lastDate.day, 0, 0, 0, 0)
	    elif cycle == 10080:
	        if len(oldDatas) > 0:
	            firstDate = getDateNum(1970, 1, 5, 0, 0, 0, 0)
	            lWeeks = ((oldDatas[len(oldDatas) - 1].date - firstDate) // 86400 + 1) // 7
	            weeks = ((latestData.date - firstDate) // 86400 + 1) // 7
	            isNextCycle = weeks > lWeeks
	    elif cycle == 43200:
	        if len(oldDatas) > 0:
	            lastDate = numToDate(oldDatas[len(oldDatas) - 1].date)
	            isNextCycle = newDate.year * 12 + newDate.month != lastDate.year * 12 + lastDate.month
	    elif cycle == 129600:
	        if len(oldDatas) > 0:
	            lastDate = numToDate(oldDatas[len(oldDatas) - 1].date)
	            isNextCycle = newDate.year * 4 + getSeason(newDate.month) != lastDate.year * 4 + getSeason(lastDate.month)
	    elif cycle == 259200:
	        if len(oldDatas) > 0:
	            lastDate = numToDate(oldDatas[len(oldDatas) - 1].date)
	            isNextCycle = newDate.year * 2 + (newDate.month // 6) != lastDate.year * 2 + (lastDate.month // 6)
	    elif cycle == 518400:
	        if len(oldDatas) > 0:
	            lastDate = numToDate(oldDatas[len(oldDatas) - 1].date)
	            isNextCycle = newDate.year != lastDate.year
    
	if isNextCycle:
	    newCycleData = SecurityData()
	    newCycleData.date = latestData.date
	    newCycleData.open = latestData.close
	    newCycleData.high = latestData.close
	    newCycleData.low = latestData.close
	    newCycleData.close = latestData.close
	    newCycleData.volume = latestData.volume - tickDataCache.lastVolume
	    newCycleData.amount = latestData.amount - tickDataCache.lastAmount
	    oldDatas.append(newCycleData)
	else:
	    if len(oldDatas) > 0:
	        lastCycleData = oldDatas[len(oldDatas) - 1]
	        if dCycle == 0:
	            thisDate = getDateNum(newDate.year, newDate.month, newDate.day, newDate.hour, newDate.minute, 0, 0)
	            for data in oldDatas:
	                if data.date == thisDate:
	                    if data.open == 0:
	                        data.open = latestData.open
	                    lastCycleData = data
	                    break
	        lastCycleData.close = latestData.close
	        if lastCycleData.high < latestData.close:
	            lastCycleData.high = latestData.close
	        if lastCycleData.low > latestData.close:
	            lastCycleData.low = latestData.close
	        lastCycleData.amount += latestData.amount - tickDataCache.lastAmount
	        lastCycleData.volume += latestData.volume - tickDataCache.lastVolume
    
	tickDataCache.code = code
	tickDataCache.lastAmount = latestData.amount
	tickDataCache.lastDate = latestData.date
	tickDataCache.lastVolume = latestData.volume

#创建一个存储调整因子的Map
factorsMap = {}

def fq_price_func(price, factor):
	""" 前复权价格计算函数
	 price 股票价格
	 factor 调整因子
	 @returns 调整后的价格"""
	cash_bt = factor.f1
	bonus_shr = factor.f3
	allot_pct = factor.f4
	allot_price = factor.f2
	return (10.0 * price - cash_bt + allot_pct * allot_price) / (10.0 + allot_pct + bonus_shr)

def fq_price_func2(price, factor):
	""" 后复权价格计算函数
	 price 股票价格
	 factor 调整因子
	 @returns 调整后的价格"""
	cash_bt = factor.f1
	bonus_shr = factor.f3
	allot_pct = factor.f4
	allot_price = factor.f2
	return (price * (10.0 + allot_pct + bonus_shr) - allot_pct * allot_price + cash_bt) / 10.0

def convertXdrBeforePrice(kd, trade_date, factor):
	""" 转换前复权
	 code 股票代码
	 kd 数据
	 trade_date 交易日期
	 factor 调整因子数组"""
	size = len(factor)
	if size > 0:
	    pos = 0
	    date = kd.date
	    if kd.date < factor[len(factor) - 1].dwDate:
	        for i in range(size):
	            if trade_date > 0 and trade_date < factor[i].dwDate:
	                continue
	            pos = i
	            if date < factor[i].dwDate:
	                break
	        for i in range(pos, size):
	            if trade_date > 0 and trade_date < factor[i].dwDate:
	                continue
	            kd.open = fq_price_func(kd.open, factor[i])
	            kd.high = fq_price_func(kd.high, factor[i])
	            kd.low = fq_price_func(kd.low, factor[i])
	            kd.close = fq_price_func(kd.close, factor[i])

def convertXdrAfterPrice(kd, trade_date, factor):
	""" 转换后复权
	 code 股票代码
	 kd 数据
	 trade_date 交易日期
	 factor 调整因子数组"""
	size = len(factor)
	if size > 0:
	    date = kd.date
	    factors = []
	    for i in range(size):
	        if date < factor[i].dwDate:
	            break
	        else:
	            factors.insert(0, factor[i])
	    for i in range(len(factors)):
	        kd.open = fq_price_func2(kd.open, factors[i])
	        kd.high = fq_price_func2(kd.high, factors[i])
	        kd.low = fq_price_func2(kd.low, factors[i])
	        kd.close = fq_price_func2(kd.close, factors[i])

def convertXdr(code, rights_offering, datas):
	""" 转换XDR
	 code 股票代码
	 rights_offering 权利发行类型
	 datas 数据数组"""
	if code in factorsMap:
	    factor = factorsMap[code]
	    datas_size = len(datas)
	    if datas_size > 0:
	        trade_date = datas[len(datas) - 1].date
	        for kd in datas:
	            if rights_offering == 1:
	                convertXdrBeforePrice(kd, trade_date, factor)
	            elif rights_offering == 2:
	                convertXdrAfterPrice(kd, trade_date, factor)
