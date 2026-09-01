-- SPDX-FileCopyrightText: Copyright (c) Microsoft Corporation
-- SPDX-License-Identifier: MIT
--
-- Ported to PostgreSQL from Microsoft's WideWorldImporters sample database
-- (https://github.com/microsoft/sql-server-samples), used under the MIT License.

-- Sequences (mirrors MSSQL Sequences schema)
CREATE SCHEMA IF NOT EXISTS Sequences;
CREATE SEQUENCE IF NOT EXISTS Sequences.BuyingGroupID INCREMENT BY 1 START WITH 3;
SELECT setval('Sequences.BuyingGroupID', 3, true);
CREATE SEQUENCE IF NOT EXISTS Sequences.CityID INCREMENT BY 1 START WITH 38187;
SELECT setval('Sequences.CityID', 38187, true);
CREATE SEQUENCE IF NOT EXISTS Sequences.ColorID INCREMENT BY 1 START WITH 37;
SELECT setval('Sequences.ColorID', 37, true);
CREATE SEQUENCE IF NOT EXISTS Sequences.CountryID INCREMENT BY 1 START WITH 242;
SELECT setval('Sequences.CountryID', 242, true);
CREATE SEQUENCE IF NOT EXISTS Sequences.CustomerCategoryID INCREMENT BY 1 START WITH 9;
SELECT setval('Sequences.CustomerCategoryID', 9, true);
CREATE SEQUENCE IF NOT EXISTS Sequences.CustomerID INCREMENT BY 1 START WITH 1062;
SELECT setval('Sequences.CustomerID', 1062, true);
CREATE SEQUENCE IF NOT EXISTS Sequences.DeliveryMethodID INCREMENT BY 1 START WITH 11;
SELECT setval('Sequences.DeliveryMethodID', 11, true);
CREATE SEQUENCE IF NOT EXISTS Sequences.InvoiceID INCREMENT BY 1 START WITH 70511;
SELECT setval('Sequences.InvoiceID', 70511, true);
CREATE SEQUENCE IF NOT EXISTS Sequences.InvoiceLineID INCREMENT BY 1 START WITH 228266;
SELECT setval('Sequences.InvoiceLineID', 228266, true);
CREATE SEQUENCE IF NOT EXISTS Sequences.OrderID INCREMENT BY 1 START WITH 73596;
SELECT setval('Sequences.OrderID', 73596, true);
CREATE SEQUENCE IF NOT EXISTS Sequences.OrderLineID INCREMENT BY 1 START WITH 231413;
SELECT setval('Sequences.OrderLineID', 231413, true);
CREATE SEQUENCE IF NOT EXISTS Sequences.PackageTypeID INCREMENT BY 1 START WITH 15;
SELECT setval('Sequences.PackageTypeID', 15, true);
CREATE SEQUENCE IF NOT EXISTS Sequences.PaymentMethodID INCREMENT BY 1 START WITH 5;
SELECT setval('Sequences.PaymentMethodID', 5, true);
CREATE SEQUENCE IF NOT EXISTS Sequences.PersonID INCREMENT BY 1 START WITH 3262;
SELECT setval('Sequences.PersonID', 3262, true);
CREATE SEQUENCE IF NOT EXISTS Sequences.PurchaseOrderID INCREMENT BY 1 START WITH 2075;
SELECT setval('Sequences.PurchaseOrderID', 2075, true);
CREATE SEQUENCE IF NOT EXISTS Sequences.PurchaseOrderLineID INCREMENT BY 1 START WITH 8368;
SELECT setval('Sequences.PurchaseOrderLineID', 8368, true);
CREATE SEQUENCE IF NOT EXISTS Sequences.SpecialDealID INCREMENT BY 1 START WITH 3;
SELECT setval('Sequences.SpecialDealID', 3, true);
CREATE SEQUENCE IF NOT EXISTS Sequences.StateProvinceID INCREMENT BY 1 START WITH 54;
SELECT setval('Sequences.StateProvinceID', 54, true);
CREATE SEQUENCE IF NOT EXISTS Sequences.StockGroupID INCREMENT BY 1 START WITH 11;
SELECT setval('Sequences.StockGroupID', 11, true);
CREATE SEQUENCE IF NOT EXISTS Sequences.StockItemID INCREMENT BY 1 START WITH 228;
SELECT setval('Sequences.StockItemID', 228, true);
CREATE SEQUENCE IF NOT EXISTS Sequences.StockItemStockGroupID INCREMENT BY 1 START WITH 443;
SELECT setval('Sequences.StockItemStockGroupID', 443, true);
CREATE SEQUENCE IF NOT EXISTS Sequences.SupplierCategoryID INCREMENT BY 1 START WITH 10;
SELECT setval('Sequences.SupplierCategoryID', 10, true);
CREATE SEQUENCE IF NOT EXISTS Sequences.SupplierID INCREMENT BY 1 START WITH 14;
SELECT setval('Sequences.SupplierID', 14, true);
CREATE SEQUENCE IF NOT EXISTS Sequences.SystemParameterID INCREMENT BY 1 START WITH 2;
SELECT setval('Sequences.SystemParameterID', 2, true);
CREATE SEQUENCE IF NOT EXISTS Sequences.TransactionID INCREMENT BY 1 START WITH 336253;
SELECT setval('Sequences.TransactionID', 336253, true);
CREATE SEQUENCE IF NOT EXISTS Sequences.TransactionTypeID INCREMENT BY 1 START WITH 14;
SELECT setval('Sequences.TransactionTypeID', 14, true);
