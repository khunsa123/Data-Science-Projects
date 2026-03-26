install.packages("forecast")
library(tseries)
pldata1=read.csv("C:/Users/khuns/OneDrive/Documents/R/pollutionData190100.csv")
pldata1_train_ts<-ts(pldata1$particullate_matter[2:10541],frequency =288 )
pldata1_test_ts<-ts(pldata1$particullate_matter[10542:17568],frequency =288 )
plot.ts(pldata1_train_ts)
plot.ts(pldata1_test_ts)

# ARIMA MODEL

pldata1_train_ts_diff1<-diff(pldata1_train_ts,differences = 1)
adf.test(pldata1_train_ts_diff1)
plot.ts(pldata1_train_ts_diff1)

# value of d= 1
acf(pldata1_train_ts_diff1,lag.max = 40)
acf(pldata1_train_ts_diff1,lag.max = 40,plot = FALSE)

# value of p = 0
pacf(pldata1_train_ts_diff1,lag.max = 40)
pacf(pldata1_train_ts_diff1,lag.max = 40,plot = FALSE)

#value of q =1

pldata1_train_ts_arima<-arima(pldata1_train_ts,order = c(1,0,1))
pldata1_train_ts_arima

#Time series forecasting
library("forecast")
pldata1_train_ts_forecast<-forecast(pldata1_train_ts_arima,h=150)
pldata1_train_ts_forecast

plot(forecast(pldata1_train_ts_forecast,h=150),shadecols=c("#ff0000","#D5DBFF"),fcol="#ff0000")
acf(pldata1_train_ts_forecast$residuals,lag.max = 20)
Box.test(pldata1_train_ts_forecast$residuals,lag = 20,type = "Ljung-Box")
