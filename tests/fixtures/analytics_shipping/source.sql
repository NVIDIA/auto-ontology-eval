-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
-- All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

PRAGMA foreign_keys = ON;

CREATE TABLE orders (
    id INTEGER PRIMARY KEY NOT NULL,
    region TEXT NOT NULL,
    order_date TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('paid', 'cancelled')),
    net_cents INTEGER NOT NULL CHECK (net_cents >= 0)
);
CREATE TABLE shipments (
    id INTEGER PRIMARY KEY NOT NULL,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    shipped_date TEXT NOT NULL
);

INSERT INTO orders VALUES
    (101, 'West', '2026-10-01', 'paid',      12500),
    (102, 'West', '2026-10-04', 'paid',       7500),
    (103, 'East', '2026-10-03', 'paid',      20000),
    (104, 'West', '2026-10-15', 'cancelled',  9900),
    (105, 'West', '2026-09-30', 'paid',       3000),
    (106, 'East', '2026-10-21', 'paid',       5000),
    (107, 'West', '2026-11-01', 'paid',      11000);

INSERT INTO shipments VALUES
    (1, 101, '2026-10-03'),
    (2, 101, '2026-10-04'),
    (3, 103, '2026-11-02'),
    (4, 104, '2026-10-16'),
    (5, 105, '2026-10-01');
