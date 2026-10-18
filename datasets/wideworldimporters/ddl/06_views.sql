-- SPDX-FileCopyrightText: Copyright (c) Microsoft Corporation
-- SPDX-License-Identifier: MIT
--
-- Ported to PostgreSQL from Microsoft's WideWorldImporters sample database
-- (https://github.com/microsoft/sql-server-samples), used under the MIT License.

-- Views (ported from MSSQL Website.* — see migrate.py phase_schema()).
--
-- The Website.VehicleTemperatures view in MSSQL uses DECOMPRESS() to inflate
-- a gzip-compressed varbinary column. Postgres has no equivalent built-in.
-- Since the WWI seed dataset has IsCompressed = 0 for every row, the
-- decompression branch is unreachable in practice; we stub it as a
-- placeholder so the view stays one-to-one structurally.

CREATE OR REPLACE VIEW Website.Customers AS
SELECT s.CustomerID,
       s.CustomerName,
       sc.CustomerCategoryName,
       pp.FullName  AS PrimaryContact,
       ap.FullName  AS AlternateContact,
       s.PhoneNumber,
       s.FaxNumber,
       bg.BuyingGroupName,
       s.WebsiteURL,
       dm.DeliveryMethodName AS DeliveryMethod,
       c.CityName   AS CityName,
       s.DeliveryLocation AS DeliveryLocation,
       s.DeliveryRun,
       s.RunPosition
FROM      Sales.Customers            AS s
LEFT JOIN Sales.CustomerCategories   AS sc ON s.CustomerCategoryID     = sc.CustomerCategoryID
LEFT JOIN Application.People         AS pp ON s.PrimaryContactPersonID = pp.PersonID
LEFT JOIN Application.People         AS ap ON s.AlternateContactPersonID = ap.PersonID
LEFT JOIN Sales.BuyingGroups         AS bg ON s.BuyingGroupID          = bg.BuyingGroupID
LEFT JOIN Application.DeliveryMethods AS dm ON s.DeliveryMethodID      = dm.DeliveryMethodID
LEFT JOIN Application.Cities         AS c  ON s.DeliveryCityID         = c.CityID;


CREATE OR REPLACE VIEW Website.Suppliers AS
SELECT s.SupplierID,
       s.SupplierName,
       sc.SupplierCategoryName,
       pp.FullName AS PrimaryContact,
       ap.FullName AS AlternateContact,
       s.PhoneNumber,
       s.FaxNumber,
       s.WebsiteURL,
       dm.DeliveryMethodName AS DeliveryMethod,
       c.CityName  AS CityName,
       s.DeliveryLocation AS DeliveryLocation,
       s.SupplierReference
FROM      Purchasing.Suppliers           AS s
LEFT JOIN Purchasing.SupplierCategories  AS sc ON s.SupplierCategoryID     = sc.SupplierCategoryID
LEFT JOIN Application.People             AS pp ON s.PrimaryContactPersonID = pp.PersonID
LEFT JOIN Application.People             AS ap ON s.AlternateContactPersonID = ap.PersonID
LEFT JOIN Application.DeliveryMethods    AS dm ON s.DeliveryMethodID       = dm.DeliveryMethodID
LEFT JOIN Application.Cities             AS c  ON s.DeliveryCityID         = c.CityID;


CREATE OR REPLACE VIEW Website.VehicleTemperatures AS
SELECT vt.VehicleTemperatureID,
       vt.VehicleRegistration,
       vt.ChillerSensorNumber,
       vt.RecordedWhen,
       vt.Temperature,
       CASE WHEN vt.IsCompressed <> false
            THEN '<compressed payload — DECOMPRESS not available in Postgres>'
            ELSE vt.FullSensorData
       END AS FullSensorData
FROM Warehouse.VehicleTemperatures AS vt;
