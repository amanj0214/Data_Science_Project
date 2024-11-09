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

SELECT 
	d_year + 20,
	d_moy,
	ca_state,
	i_class,
	i_category,
	sum(ws_quantity),
	sum(wr_return_quantity),
	-- wr_net_loss = wr_return_amt_inc_tax − (wr_refunded_cash + wr_reversed_charge + wr_account_credit) + wr_return_ship_cost + wr_fee
	sum(wr_net_loss)
FROM web_sales ws 
JOIN item i ON ws.ws_item_sk = i.i_item_sk 
JOIN date_dim dd ON dd.d_date_sk = ws.ws_sold_date_sk 
JOIN web_returns wr ON wr.wr_order_number = ws.ws_order_number AND wr.wr_item_sk = ws.ws_item_sk 
JOIN customer_address ca ON wr.wr_returning_addr_sk = ca.ca_address_sk 
WHERE 
	1=1
	AND d_year IS NOT NULL
	AND d_moy IS NOT NULL 
	AND i_class != 'None'
	AND i_category != 'None' 
GROUP BY 
	d_year,
	d_moy,
	ca_state,
	i_class,
	i_category
HAVING sum(wr_net_loss) > avg(wr_net_loss)

SELECT ca_state, count(1) FROM customer_address ca GROUP BY ca_state LIMIT 10

SELECT *
FROM item