-- Custom reporting tables, materialized from the migrated WWI data.
-- Re-runnable: each output table is dropped before creation.
--
-- Notes vs. the original MSSQL draft:
--   * Source tables/columns are CamelCase in Postgres (migrated as quoted
--     identifiers), so every reference is double-quoted.
--   * MSSQL YEAR()/MONTH()      -> EXTRACT(YEAR|MONTH FROM ...)::int
--   * MSSQL DATEDIFF(DAY, a, b) -> (b::date - a::date)
--   * Output schema/tables use unquoted lowercase, idiomatic for Postgres.

CREATE SCHEMA IF NOT EXISTS reports;


DROP TABLE IF EXISTS reports.monthly_inventory_turnover;
CREATE TABLE reports.monthly_inventory_turnover AS
SELECT EXTRACT(YEAR  FROM st.TransactionOccurredWhen)::int AS year,
       EXTRACT(MONTH FROM st.TransactionOccurredWhen)::int AS month,
       si.StockItemID   AS stockitemid,
       si.StockItemName AS stockitemname,
       SUM(st.Quantity) AS total_quantity
FROM Warehouse.StockItemTransactions st
JOIN Warehouse.StockItems si ON st.StockItemID = si.StockItemID
GROUP BY 1, 2, si.StockItemID, si.StockItemName;


DROP TABLE IF EXISTS reports.customer_retention;
CREATE TABLE reports.customer_retention AS
SELECT c.CustomerID     AS customerid,
       c.CustomerName   AS customername,
       MIN(o.OrderDate) AS first_order_date,
       MAX(o.OrderDate) AS last_order_date,
       COUNT(o.OrderID) AS order_count
FROM Sales.Customers c
JOIN Sales.Orders    o ON c.CustomerID = o.CustomerID
GROUP BY c.CustomerID, c.CustomerName;


DROP TABLE IF EXISTS reports.supplier_transaction_summary;
CREATE TABLE reports.supplier_transaction_summary AS
SELECT s.SupplierID   AS supplierid,
       s.SupplierName AS suppliername,
       COUNT(st.SupplierTransactionID) AS transaction_count
FROM Purchasing.Suppliers s
JOIN Purchasing.SupplierTransactions st ON s.SupplierID = st.SupplierID
GROUP BY s.SupplierID, s.SupplierName;


DROP TABLE IF EXISTS reports.delayed_orders;
CREATE TABLE reports.delayed_orders AS
SELECT o.OrderID               AS orderid,
       o.OrderDate             AS orderdate,
       o.PickingCompletedWhen  AS pickingcompletedwhen,
       (o.PickingCompletedWhen::date - o.OrderDate::date) AS delivery_delay
FROM Sales.Orders o
WHERE o.PickingCompletedWhen > o.OrderDate;


DROP TABLE IF EXISTS reports.top_selling_products;
CREATE TABLE reports.top_selling_products AS
SELECT si.StockItemID        AS stockitemid,
       si.StockItemName      AS stockitemname,
       SUM(il.Quantity)      AS total_quantity,
       SUM(il.ExtendedPrice) AS total_revenue
FROM Warehouse.StockItems si
JOIN Sales.InvoiceLines   il ON si.StockItemID = il.StockItemID
GROUP BY si.StockItemID, si.StockItemName
ORDER BY total_revenue DESC;


DROP TABLE IF EXISTS reports.stock_item_transaction_summary;
CREATE TABLE reports.stock_item_transaction_summary AS
SELECT si.StockItemID   AS stockitemid,
       si.StockItemName AS stockitemname,
       SUM(st.Quantity) AS total_quantity,
       COUNT(st.StockItemTransactionID) AS transaction_count
FROM Warehouse.StockItems si
JOIN Warehouse.StockItemTransactions st ON si.StockItemID = st.StockItemID
GROUP BY si.StockItemID, si.StockItemName;


DROP TABLE IF EXISTS reports.supplier_purchase_order_summaries;
CREATE TABLE reports.supplier_purchase_order_summaries AS
SELECT s.SupplierID               AS supplierid,
       s.SupplierName             AS suppliername,
       COUNT(po.PurchaseOrderID)  AS total_orders,
       SUM(pol.ExpectedUnitPricePerOuter * pol.ReceivedOuters) AS total_purchased
FROM Purchasing.Suppliers           s
JOIN Purchasing.PurchaseOrders      po  ON s.SupplierID      = po.SupplierID
JOIN Purchasing.PurchaseOrderLines  pol ON po.PurchaseOrderID = pol.PurchaseOrderID
GROUP BY s.SupplierID, s.SupplierName;


DROP TABLE IF EXISTS reports.customer_order_summaries;
CREATE TABLE reports.customer_order_summaries AS
SELECT c.CustomerID      AS customerid,
       c.CustomerName    AS customername,
       COUNT(o.OrderID)  AS total_orders
FROM Sales.Customers c
JOIN Sales.Orders    o ON c.CustomerID = o.CustomerID
GROUP BY c.CustomerID, c.CustomerName;


DROP TABLE IF EXISTS reports.monthly_sales_aggregate;
CREATE TABLE reports.monthly_sales_aggregate AS
SELECT EXTRACT(YEAR  FROM LastEditedWhen)::int AS year,
       EXTRACT(MONTH FROM LastEditedWhen)::int AS month,
       SUM(ExtendedPrice) AS total_sales
FROM Sales.InvoiceLines
GROUP BY 1, 2;
