-- SPDX-FileCopyrightText: Copyright (c) Microsoft Corporation
-- SPDX-License-Identifier: MIT
--
-- Ported to PostgreSQL from Microsoft's WideWorldImporters sample database
-- (https://github.com/microsoft/sql-server-samples), used under the MIT License.

-- Tables (FKs deferred to 05_fkeys.sql)

CREATE TABLE Application.Cities (
    CityID integer NOT NULL,
    CityName varchar(50) NOT NULL,
    StateProvinceID integer NOT NULL,
    Location text,
    LatestRecordedPopulation bigint,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL,
    PRIMARY KEY (CityID)
);

CREATE TABLE Application.Cities_Archive (
    CityID integer NOT NULL,
    CityName varchar(50) NOT NULL,
    StateProvinceID integer NOT NULL,
    Location text,
    LatestRecordedPopulation bigint,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL
);

CREATE TABLE Application.Countries (
    CountryID integer NOT NULL,
    CountryName varchar(60) NOT NULL,
    FormalName varchar(60) NOT NULL,
    IsoAlpha3Code varchar(3),
    IsoNumericCode integer,
    CountryType varchar(20),
    LatestRecordedPopulation bigint,
    Continent varchar(30) NOT NULL,
    Region varchar(30) NOT NULL,
    Subregion varchar(30) NOT NULL,
    Border text,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL,
    PRIMARY KEY (CountryID)
);

CREATE TABLE Application.Countries_Archive (
    CountryID integer NOT NULL,
    CountryName varchar(60) NOT NULL,
    FormalName varchar(60) NOT NULL,
    IsoAlpha3Code varchar(3),
    IsoNumericCode integer,
    CountryType varchar(20),
    LatestRecordedPopulation bigint,
    Continent varchar(30) NOT NULL,
    Region varchar(30) NOT NULL,
    Subregion varchar(30) NOT NULL,
    Border text,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL
);

CREATE TABLE Application.DeliveryMethods (
    DeliveryMethodID integer NOT NULL,
    DeliveryMethodName varchar(50) NOT NULL,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL,
    PRIMARY KEY (DeliveryMethodID)
);

CREATE TABLE Application.DeliveryMethods_Archive (
    DeliveryMethodID integer NOT NULL,
    DeliveryMethodName varchar(50) NOT NULL,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL
);

CREATE TABLE Application.PaymentMethods (
    PaymentMethodID integer NOT NULL,
    PaymentMethodName varchar(50) NOT NULL,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL,
    PRIMARY KEY (PaymentMethodID)
);

CREATE TABLE Application.PaymentMethods_Archive (
    PaymentMethodID integer NOT NULL,
    PaymentMethodName varchar(50) NOT NULL,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL
);

CREATE TABLE Application.People (
    PersonID integer NOT NULL,
    FullName varchar(50) NOT NULL,
    PreferredName varchar(50) NOT NULL,
    SearchName varchar(101) NOT NULL,
    IsPermittedToLogon boolean NOT NULL,
    LogonName varchar(50),
    IsExternalLogonProvider boolean NOT NULL,
    HashedPassword bytea,
    IsSystemUser boolean NOT NULL,
    IsEmployee boolean NOT NULL,
    IsSalesperson boolean NOT NULL,
    UserPreferences text,
    PhoneNumber varchar(20),
    FaxNumber varchar(20),
    EmailAddress varchar(256),
    Photo bytea,
    CustomFields text,
    OtherLanguages text,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL,
    PRIMARY KEY (PersonID)
);

CREATE TABLE Application.People_Archive (
    PersonID integer NOT NULL,
    FullName varchar(50) NOT NULL,
    PreferredName varchar(50) NOT NULL,
    SearchName varchar(101) NOT NULL,
    IsPermittedToLogon boolean NOT NULL,
    LogonName varchar(50),
    IsExternalLogonProvider boolean NOT NULL,
    HashedPassword bytea,
    IsSystemUser boolean NOT NULL,
    IsEmployee boolean NOT NULL,
    IsSalesperson boolean NOT NULL,
    UserPreferences text,
    PhoneNumber varchar(20),
    FaxNumber varchar(20),
    EmailAddress varchar(256),
    Photo bytea,
    CustomFields text,
    OtherLanguages text,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL
);

CREATE TABLE Application.StateProvinces (
    StateProvinceID integer NOT NULL,
    StateProvinceCode varchar(5) NOT NULL,
    StateProvinceName varchar(50) NOT NULL,
    CountryID integer NOT NULL,
    SalesTerritory varchar(50) NOT NULL,
    Border text,
    LatestRecordedPopulation bigint,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL,
    PRIMARY KEY (StateProvinceID)
);

CREATE TABLE Application.StateProvinces_Archive (
    StateProvinceID integer NOT NULL,
    StateProvinceCode varchar(5) NOT NULL,
    StateProvinceName varchar(50) NOT NULL,
    CountryID integer NOT NULL,
    SalesTerritory varchar(50) NOT NULL,
    Border text,
    LatestRecordedPopulation bigint,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL
);

CREATE TABLE Application.SystemParameters (
    SystemParameterID integer NOT NULL,
    DeliveryAddressLine1 varchar(60) NOT NULL,
    DeliveryAddressLine2 varchar(60),
    DeliveryCityID integer NOT NULL,
    DeliveryPostalCode varchar(10) NOT NULL,
    DeliveryLocation text NOT NULL,
    PostalAddressLine1 varchar(60) NOT NULL,
    PostalAddressLine2 varchar(60),
    PostalCityID integer NOT NULL,
    PostalPostalCode varchar(10) NOT NULL,
    ApplicationSettings text NOT NULL,
    LastEditedBy integer NOT NULL,
    LastEditedWhen timestamp NOT NULL,
    PRIMARY KEY (SystemParameterID)
);

CREATE TABLE Application.TransactionTypes (
    TransactionTypeID integer NOT NULL,
    TransactionTypeName varchar(50) NOT NULL,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL,
    PRIMARY KEY (TransactionTypeID)
);

CREATE TABLE Application.TransactionTypes_Archive (
    TransactionTypeID integer NOT NULL,
    TransactionTypeName varchar(50) NOT NULL,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL
);

CREATE TABLE Purchasing.PurchaseOrderLines (
    PurchaseOrderLineID integer NOT NULL,
    PurchaseOrderID integer NOT NULL,
    StockItemID integer NOT NULL,
    OrderedOuters integer NOT NULL,
    Description varchar(100) NOT NULL,
    ReceivedOuters integer NOT NULL,
    PackageTypeID integer NOT NULL,
    ExpectedUnitPricePerOuter numeric(18,2),
    LastReceiptDate date,
    IsOrderLineFinalized boolean NOT NULL,
    LastEditedBy integer NOT NULL,
    LastEditedWhen timestamp NOT NULL,
    PRIMARY KEY (PurchaseOrderLineID)
);

CREATE TABLE Purchasing.PurchaseOrders (
    PurchaseOrderID integer NOT NULL,
    SupplierID integer NOT NULL,
    OrderDate date NOT NULL,
    DeliveryMethodID integer NOT NULL,
    ContactPersonID integer NOT NULL,
    ExpectedDeliveryDate date,
    SupplierReference varchar(20),
    IsOrderFinalized boolean NOT NULL,
    Comments text,
    InternalComments text,
    LastEditedBy integer NOT NULL,
    LastEditedWhen timestamp NOT NULL,
    PRIMARY KEY (PurchaseOrderID)
);

CREATE TABLE Purchasing.SupplierCategories (
    SupplierCategoryID integer NOT NULL,
    SupplierCategoryName varchar(50) NOT NULL,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL,
    PRIMARY KEY (SupplierCategoryID)
);

CREATE TABLE Purchasing.SupplierCategories_Archive (
    SupplierCategoryID integer NOT NULL,
    SupplierCategoryName varchar(50) NOT NULL,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL
);

CREATE TABLE Purchasing.Suppliers (
    SupplierID integer NOT NULL,
    SupplierName varchar(100) NOT NULL,
    SupplierCategoryID integer NOT NULL,
    PrimaryContactPersonID integer NOT NULL,
    AlternateContactPersonID integer NOT NULL,
    DeliveryMethodID integer,
    DeliveryCityID integer NOT NULL,
    PostalCityID integer NOT NULL,
    SupplierReference varchar(20),
    BankAccountName varchar(50),
    BankAccountBranch varchar(50),
    BankAccountCode varchar(20),
    BankAccountNumber varchar(20),
    BankInternationalCode varchar(20),
    PaymentDays integer NOT NULL,
    InternalComments text,
    PhoneNumber varchar(20) NOT NULL,
    FaxNumber varchar(20) NOT NULL,
    WebsiteURL varchar(256) NOT NULL,
    DeliveryAddressLine1 varchar(60) NOT NULL,
    DeliveryAddressLine2 varchar(60),
    DeliveryPostalCode varchar(10) NOT NULL,
    DeliveryLocation text,
    PostalAddressLine1 varchar(60) NOT NULL,
    PostalAddressLine2 varchar(60),
    PostalPostalCode varchar(10) NOT NULL,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL,
    PRIMARY KEY (SupplierID)
);

CREATE TABLE Purchasing.Suppliers_Archive (
    SupplierID integer NOT NULL,
    SupplierName varchar(100) NOT NULL,
    SupplierCategoryID integer NOT NULL,
    PrimaryContactPersonID integer NOT NULL,
    AlternateContactPersonID integer NOT NULL,
    DeliveryMethodID integer,
    DeliveryCityID integer NOT NULL,
    PostalCityID integer NOT NULL,
    SupplierReference varchar(20),
    BankAccountName varchar(50),
    BankAccountBranch varchar(50),
    BankAccountCode varchar(20),
    BankAccountNumber varchar(20),
    BankInternationalCode varchar(20),
    PaymentDays integer NOT NULL,
    InternalComments text,
    PhoneNumber varchar(20) NOT NULL,
    FaxNumber varchar(20) NOT NULL,
    WebsiteURL varchar(256) NOT NULL,
    DeliveryAddressLine1 varchar(60) NOT NULL,
    DeliveryAddressLine2 varchar(60),
    DeliveryPostalCode varchar(10) NOT NULL,
    DeliveryLocation text,
    PostalAddressLine1 varchar(60) NOT NULL,
    PostalAddressLine2 varchar(60),
    PostalPostalCode varchar(10) NOT NULL,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL
);

CREATE TABLE Purchasing.SupplierTransactions (
    SupplierTransactionID integer NOT NULL,
    SupplierID integer NOT NULL,
    TransactionTypeID integer NOT NULL,
    PurchaseOrderID integer,
    PaymentMethodID integer,
    SupplierInvoiceNumber varchar(20),
    TransactionDate date NOT NULL,
    AmountExcludingTax numeric(18,2) NOT NULL,
    TaxAmount numeric(18,2) NOT NULL,
    TransactionAmount numeric(18,2) NOT NULL,
    OutstandingBalance numeric(18,2) NOT NULL,
    FinalizationDate date,
    IsFinalized boolean,
    LastEditedBy integer NOT NULL,
    LastEditedWhen timestamp NOT NULL,
    PRIMARY KEY (SupplierTransactionID)
);

CREATE TABLE Sales.BuyingGroups (
    BuyingGroupID integer NOT NULL,
    BuyingGroupName varchar(50) NOT NULL,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL,
    PRIMARY KEY (BuyingGroupID)
);

CREATE TABLE Sales.BuyingGroups_Archive (
    BuyingGroupID integer NOT NULL,
    BuyingGroupName varchar(50) NOT NULL,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL
);

CREATE TABLE Sales.CustomerCategories (
    CustomerCategoryID integer NOT NULL,
    CustomerCategoryName varchar(50) NOT NULL,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL,
    PRIMARY KEY (CustomerCategoryID)
);

CREATE TABLE Sales.CustomerCategories_Archive (
    CustomerCategoryID integer NOT NULL,
    CustomerCategoryName varchar(50) NOT NULL,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL
);

CREATE TABLE Sales.Customers (
    CustomerID integer NOT NULL,
    CustomerName varchar(100) NOT NULL,
    BillToCustomerID integer NOT NULL,
    CustomerCategoryID integer NOT NULL,
    BuyingGroupID integer,
    PrimaryContactPersonID integer NOT NULL,
    AlternateContactPersonID integer,
    DeliveryMethodID integer NOT NULL,
    DeliveryCityID integer NOT NULL,
    PostalCityID integer NOT NULL,
    CreditLimit numeric(18,2),
    AccountOpenedDate date NOT NULL,
    StandardDiscountPercentage numeric(18,3) NOT NULL,
    IsStatementSent boolean NOT NULL,
    IsOnCreditHold boolean NOT NULL,
    PaymentDays integer NOT NULL,
    PhoneNumber varchar(20) NOT NULL,
    FaxNumber varchar(20) NOT NULL,
    DeliveryRun varchar(5),
    RunPosition varchar(5),
    WebsiteURL varchar(256) NOT NULL,
    DeliveryAddressLine1 varchar(60) NOT NULL,
    DeliveryAddressLine2 varchar(60),
    DeliveryPostalCode varchar(10) NOT NULL,
    DeliveryLocation text,
    PostalAddressLine1 varchar(60) NOT NULL,
    PostalAddressLine2 varchar(60),
    PostalPostalCode varchar(10) NOT NULL,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL,
    PRIMARY KEY (CustomerID)
);

CREATE TABLE Sales.Customers_Archive (
    CustomerID integer NOT NULL,
    CustomerName varchar(100) NOT NULL,
    BillToCustomerID integer NOT NULL,
    CustomerCategoryID integer NOT NULL,
    BuyingGroupID integer,
    PrimaryContactPersonID integer NOT NULL,
    AlternateContactPersonID integer,
    DeliveryMethodID integer NOT NULL,
    DeliveryCityID integer NOT NULL,
    PostalCityID integer NOT NULL,
    CreditLimit numeric(18,2),
    AccountOpenedDate date NOT NULL,
    StandardDiscountPercentage numeric(18,3) NOT NULL,
    IsStatementSent boolean NOT NULL,
    IsOnCreditHold boolean NOT NULL,
    PaymentDays integer NOT NULL,
    PhoneNumber varchar(20) NOT NULL,
    FaxNumber varchar(20) NOT NULL,
    DeliveryRun varchar(5),
    RunPosition varchar(5),
    WebsiteURL varchar(256) NOT NULL,
    DeliveryAddressLine1 varchar(60) NOT NULL,
    DeliveryAddressLine2 varchar(60),
    DeliveryPostalCode varchar(10) NOT NULL,
    DeliveryLocation text,
    PostalAddressLine1 varchar(60) NOT NULL,
    PostalAddressLine2 varchar(60),
    PostalPostalCode varchar(10) NOT NULL,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL
);

CREATE TABLE Sales.CustomerTransactions (
    CustomerTransactionID integer NOT NULL,
    CustomerID integer NOT NULL,
    TransactionTypeID integer NOT NULL,
    InvoiceID integer,
    PaymentMethodID integer,
    TransactionDate date NOT NULL,
    AmountExcludingTax numeric(18,2) NOT NULL,
    TaxAmount numeric(18,2) NOT NULL,
    TransactionAmount numeric(18,2) NOT NULL,
    OutstandingBalance numeric(18,2) NOT NULL,
    FinalizationDate date,
    IsFinalized boolean,
    LastEditedBy integer NOT NULL,
    LastEditedWhen timestamp NOT NULL,
    PRIMARY KEY (CustomerTransactionID)
);

CREATE TABLE Sales.InvoiceLines (
    InvoiceLineID integer NOT NULL,
    InvoiceID integer NOT NULL,
    StockItemID integer NOT NULL,
    Description varchar(100) NOT NULL,
    PackageTypeID integer NOT NULL,
    Quantity integer NOT NULL,
    UnitPrice numeric(18,2),
    TaxRate numeric(18,3) NOT NULL,
    TaxAmount numeric(18,2) NOT NULL,
    LineProfit numeric(18,2) NOT NULL,
    ExtendedPrice numeric(18,2) NOT NULL,
    LastEditedBy integer NOT NULL,
    LastEditedWhen timestamp NOT NULL,
    PRIMARY KEY (InvoiceLineID)
);

CREATE TABLE Sales.Invoices (
    InvoiceID integer NOT NULL,
    CustomerID integer NOT NULL,
    BillToCustomerID integer NOT NULL,
    OrderID integer,
    DeliveryMethodID integer NOT NULL,
    ContactPersonID integer NOT NULL,
    AccountsPersonID integer NOT NULL,
    SalespersonPersonID integer NOT NULL,
    PackedByPersonID integer NOT NULL,
    InvoiceDate date NOT NULL,
    CustomerPurchaseOrderNumber varchar(20),
    IsCreditNote boolean NOT NULL,
    CreditNoteReason text,
    Comments text,
    DeliveryInstructions text,
    InternalComments text,
    TotalDryItems integer NOT NULL,
    TotalChillerItems integer NOT NULL,
    DeliveryRun varchar(5),
    RunPosition varchar(5),
    ReturnedDeliveryData text,
    ConfirmedDeliveryTime timestamp,
    ConfirmedReceivedBy varchar(4000),
    LastEditedBy integer NOT NULL,
    LastEditedWhen timestamp NOT NULL,
    PRIMARY KEY (InvoiceID)
);

CREATE TABLE Sales.OrderLines (
    OrderLineID integer NOT NULL,
    OrderID integer NOT NULL,
    StockItemID integer NOT NULL,
    Description varchar(100) NOT NULL,
    PackageTypeID integer NOT NULL,
    Quantity integer NOT NULL,
    UnitPrice numeric(18,2),
    TaxRate numeric(18,3) NOT NULL,
    PickedQuantity integer NOT NULL,
    PickingCompletedWhen timestamp,
    LastEditedBy integer NOT NULL,
    LastEditedWhen timestamp NOT NULL,
    PRIMARY KEY (OrderLineID)
);

CREATE TABLE Sales.Orders (
    OrderID integer NOT NULL,
    CustomerID integer NOT NULL,
    SalespersonPersonID integer NOT NULL,
    PickedByPersonID integer,
    ContactPersonID integer NOT NULL,
    BackorderOrderID integer,
    OrderDate date NOT NULL,
    ExpectedDeliveryDate date NOT NULL,
    CustomerPurchaseOrderNumber varchar(20),
    IsUndersupplyBackordered boolean NOT NULL,
    Comments text,
    DeliveryInstructions text,
    InternalComments text,
    PickingCompletedWhen timestamp,
    LastEditedBy integer NOT NULL,
    LastEditedWhen timestamp NOT NULL,
    PRIMARY KEY (OrderID)
);

CREATE TABLE Sales.SpecialDeals (
    SpecialDealID integer NOT NULL,
    StockItemID integer,
    CustomerID integer,
    BuyingGroupID integer,
    CustomerCategoryID integer,
    StockGroupID integer,
    DealDescription varchar(30) NOT NULL,
    StartDate date NOT NULL,
    EndDate date NOT NULL,
    DiscountAmount numeric(18,2),
    DiscountPercentage numeric(18,3),
    UnitPrice numeric(18,2),
    LastEditedBy integer NOT NULL,
    LastEditedWhen timestamp NOT NULL,
    PRIMARY KEY (SpecialDealID)
);

CREATE TABLE Warehouse.ColdRoomTemperatures (
    ColdRoomTemperatureID bigint NOT NULL,
    ColdRoomSensorNumber integer NOT NULL,
    RecordedWhen timestamp NOT NULL,
    Temperature numeric(10,2) NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL,
    PRIMARY KEY (ColdRoomTemperatureID)
);

CREATE TABLE Warehouse.ColdRoomTemperatures_Archive (
    ColdRoomTemperatureID bigint NOT NULL,
    ColdRoomSensorNumber integer NOT NULL,
    RecordedWhen timestamp NOT NULL,
    Temperature numeric(10,2) NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL
);

CREATE TABLE Warehouse.Colors (
    ColorID integer NOT NULL,
    ColorName varchar(20) NOT NULL,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL,
    PRIMARY KEY (ColorID)
);

CREATE TABLE Warehouse.Colors_Archive (
    ColorID integer NOT NULL,
    ColorName varchar(20) NOT NULL,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL
);

CREATE TABLE Warehouse.PackageTypes (
    PackageTypeID integer NOT NULL,
    PackageTypeName varchar(50) NOT NULL,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL,
    PRIMARY KEY (PackageTypeID)
);

CREATE TABLE Warehouse.PackageTypes_Archive (
    PackageTypeID integer NOT NULL,
    PackageTypeName varchar(50) NOT NULL,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL
);

CREATE TABLE Warehouse.StockGroups (
    StockGroupID integer NOT NULL,
    StockGroupName varchar(50) NOT NULL,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL,
    PRIMARY KEY (StockGroupID)
);

CREATE TABLE Warehouse.StockGroups_Archive (
    StockGroupID integer NOT NULL,
    StockGroupName varchar(50) NOT NULL,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL
);

CREATE TABLE Warehouse.StockItemHoldings (
    StockItemID integer NOT NULL,
    QuantityOnHand integer NOT NULL,
    BinLocation varchar(20) NOT NULL,
    LastStocktakeQuantity integer NOT NULL,
    LastCostPrice numeric(18,2) NOT NULL,
    ReorderLevel integer NOT NULL,
    TargetStockLevel integer NOT NULL,
    LastEditedBy integer NOT NULL,
    LastEditedWhen timestamp NOT NULL,
    PRIMARY KEY (StockItemID)
);

CREATE TABLE Warehouse.StockItems (
    StockItemID integer NOT NULL,
    StockItemName varchar(100) NOT NULL,
    SupplierID integer NOT NULL,
    ColorID integer,
    UnitPackageID integer NOT NULL,
    OuterPackageID integer NOT NULL,
    Brand varchar(50),
    Size varchar(20),
    LeadTimeDays integer NOT NULL,
    QuantityPerOuter integer NOT NULL,
    IsChillerStock boolean NOT NULL,
    Barcode varchar(50),
    TaxRate numeric(18,3) NOT NULL,
    UnitPrice numeric(18,2) NOT NULL,
    RecommendedRetailPrice numeric(18,2),
    TypicalWeightPerUnit numeric(18,3) NOT NULL,
    MarketingComments text,
    InternalComments text,
    Photo bytea,
    CustomFields text,
    Tags text,
    SearchDetails text NOT NULL,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL,
    PRIMARY KEY (StockItemID)
);

CREATE TABLE Warehouse.StockItems_Archive (
    StockItemID integer NOT NULL,
    StockItemName varchar(100) NOT NULL,
    SupplierID integer NOT NULL,
    ColorID integer,
    UnitPackageID integer NOT NULL,
    OuterPackageID integer NOT NULL,
    Brand varchar(50),
    Size varchar(20),
    LeadTimeDays integer NOT NULL,
    QuantityPerOuter integer NOT NULL,
    IsChillerStock boolean NOT NULL,
    Barcode varchar(50),
    TaxRate numeric(18,3) NOT NULL,
    UnitPrice numeric(18,2) NOT NULL,
    RecommendedRetailPrice numeric(18,2),
    TypicalWeightPerUnit numeric(18,3) NOT NULL,
    MarketingComments text,
    InternalComments text,
    Photo bytea,
    CustomFields text,
    Tags text,
    SearchDetails text NOT NULL,
    LastEditedBy integer NOT NULL,
    ValidFrom timestamp NOT NULL,
    ValidTo timestamp NOT NULL
);

CREATE TABLE Warehouse.StockItemStockGroups (
    StockItemStockGroupID integer NOT NULL,
    StockItemID integer NOT NULL,
    StockGroupID integer NOT NULL,
    LastEditedBy integer NOT NULL,
    LastEditedWhen timestamp NOT NULL,
    PRIMARY KEY (StockItemStockGroupID)
);

CREATE TABLE Warehouse.StockItemTransactions (
    StockItemTransactionID integer NOT NULL,
    StockItemID integer NOT NULL,
    TransactionTypeID integer NOT NULL,
    CustomerID integer,
    InvoiceID integer,
    SupplierID integer,
    PurchaseOrderID integer,
    TransactionOccurredWhen timestamp NOT NULL,
    Quantity numeric(18,3) NOT NULL,
    LastEditedBy integer NOT NULL,
    LastEditedWhen timestamp NOT NULL,
    PRIMARY KEY (StockItemTransactionID)
);

CREATE TABLE Warehouse.VehicleTemperatures (
    VehicleTemperatureID bigint NOT NULL,
    VehicleRegistration varchar(20) NOT NULL,
    ChillerSensorNumber integer NOT NULL,
    RecordedWhen timestamp NOT NULL,
    Temperature numeric(10,2) NOT NULL,
    FullSensorData varchar(1000),
    IsCompressed boolean NOT NULL,
    CompressedSensorData bytea,
    PRIMARY KEY (VehicleTemperatureID)
);
