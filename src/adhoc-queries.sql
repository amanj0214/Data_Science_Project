-- Daily WebSales? Weekly WebSales? Monthly WebSales? #Stores?
-- Shipping? ? ?
-- Q1: Get all the returned categories
select i_category, i.i_class, count(1)
from web_returns wr join item i on wr.wr_item_sk = i.i_item_sk
group by i_category, i_class


-- Q2: Get all the returns for Electronics
select *
from web_returns wr join item i on wr.wr_item_sk = i.i_item_sk
WHERE i.i_category = 'Electronics'
limit 10