# msc_project_a
MSc project Predicting stock price movements using sentiment data

# QUESTIONS FOR ALESSANDRO
1.  Should I be creating pretty looking notebooks with every command and result with  
commentary? - or just ones I deem to be important?
2. Demo video - speak to Alessandro.  
3. How do I give access to Github?  
4. in the appendix and README give instructions how to use the software.  
5. Do not include code in the main report.  
6. Explain why you do everything in the context of your own project.  
7. Bibliography  - doesn't need to be cited. Just say name of book eg. Deep Learning - Goodfellow. You can also put  AI in this section - work out how to include it.  
8. At the moment it seems more appropriate to keep making notebooks rather than python files - mainly becuase I haven't really come across any repeated tasks yet - I assume this is ok?  
9. 

# CURRENT POINT IN PROJECT:  
Move all weekend news to the next trading day - use the library 

# ORDER OF NOTEBOOKS
1. load_raw_sentiment.ipynb  
2. prep_tickers_for_bberg.ipynb  
3. filter_mktcap.ipynb  
4. get_price_data.ipynb  
5. sentiment_timestamp_adjustment.ipynb  
6. daily_sentiment_aggregation.ipynb  
7. 

Total raw file lines: 40,473,352  
Countries: ['USA' 'CAN' 'MEX']

Min date: 2000-01-01 02:56:05.000  
Max date: 2025-06-30 23:53:35.220

Country
USA    37445514  
CAN     2937242  
MEX       90596  

Filter by date > 2018-01-01... rows of US data = 8,679,192
2738 days of data (appox 2738 / 365 = 7.5 years - about from from beginning of 2018 to half way through 2025.)  

A quick bar graph showing the headline counts per month for the entire 7.5 year period.
Minium monthly number of headlines: 64987
Maximum monthly number of headlines: 141913
Average monthly number of headlines: 96435  



![img.png](img.png)  

After Filtering for market cap and turnover we were left with 1,469 securities. This is how the sentiment headlines look after this filter  

![img_4.png](img_4.png)  

Minium monthly number of headlines: 25176
Maximum monthly number of headlines: 55018
Average monthly number of headlines: 39408.96666666667


Matching up ISINs and tickers has taken much longer than expected and Bloomberg throttled my api data usage. I managed to do a signifcant  
amount of the isin and ticker matching inside BQuant as this was sandboxed to achieve a good data field for 1,911 securities to now begin the mkt_cap  
and turnover filters... 

The distribution of mkt_cap in USD billions.

![img_2.png](img_2.png)

The distribution of daily turnover in USD millions

![img_3.png](img_3.png)

Using stock market trading experience which is backed up by current research a market cap filter minimum of $1bln and a turnover  
minimum of $5m per day (Gu et al 2020) will be applied to the stock universe to ensure minimum market impact and reduction of  
trading spreads. The impact on the numbers of securities are below:  
Number of separate securities: 1911 before filters applied. 
Number of securities by mkt cap filter only  1562.  
Number of securities by turnover filter only: 1609.  
Number of securities by both filters: 1469.    

Prices downloaded from Bloomberg inside their BQNT sandbox - prices copied and pasted into .txt file.

Fix nan prices. I had to backwards fill 65 securties who's missing values happened to be before their first listing date. Backfilling from here was  
was the right fix.

# Next task was to move sentiment data from non-trading days.  
I realised there were actually three things to do:
1. Adjust/Translate time from UTC to EST (New York).
2. Move any data that occurred after 4pm NY time to the next trading day and apply the timestamp as 00:01:00. 
3. Move any data that occurred on an non-trading day (weekend or holiday) to the next trading day and apply the timestamp as 00:01:00

# Calcualte daily sentiment measures.  
1. For reasoning and structure of the 4 built measures refer to the file in .idea folder.  
2. Present some stats on these daily structures.





# Struggles / Weaknesses / Further things to develop  
1. Time it took for ticker / ISIN matching - historical data is tough to work with.  
2. API throttle. Meant lot's of copying and pasting from BQNT. Explain how you batch saved.  
3. Given we had daily closing prices we had to aggregate the sentiment to daily summaries. However, with greater detail we could have potentially  
traded on a market open for news that was out before the open - maybe this is achievable?  