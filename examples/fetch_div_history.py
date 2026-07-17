from bbg_fetch import fetch_div_yields
tickers = ['FFASIAY LX Equity']
_, _, divs_1y = fetch_div_yields(tickers=tickers)
print(divs_1y)
