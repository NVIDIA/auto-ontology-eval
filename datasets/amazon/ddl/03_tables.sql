-- Tables (FKs deferred to 05_fkeys.sql)

CREATE TABLE amazon.products (
    product_id integer NOT NULL,
    asin text,
    title text,
    description text,
    feature text,
    price text,
    rank text,
    global_category text,
    details text,
    brand_id integer,
    PRIMARY KEY (product_id)
);

CREATE TABLE amazon.brands (
    brand_id integer NOT NULL,
    brand_name text,
    PRIMARY KEY (brand_id)
);

CREATE TABLE amazon.categories (
    category_id integer NOT NULL,
    category_name text,
    PRIMARY KEY (category_id)
);

CREATE TABLE amazon.colors (
    color_id integer NOT NULL,
    color_name text,
    PRIMARY KEY (color_id)
);

CREATE TABLE amazon.product_categories (
    product_id integer NOT NULL,
    category_id integer NOT NULL,
    PRIMARY KEY (product_id, category_id)
);

CREATE TABLE amazon.product_colors (
    product_id integer NOT NULL,
    color_id integer NOT NULL,
    PRIMARY KEY (product_id, color_id)
);

CREATE TABLE amazon.also_buy (
    product_id integer NOT NULL,
    related_product_id integer NOT NULL,
    PRIMARY KEY (product_id, related_product_id)
);

CREATE TABLE amazon.also_view (
    product_id integer NOT NULL,
    related_product_id integer NOT NULL,
    PRIMARY KEY (product_id, related_product_id)
);

CREATE TABLE amazon.reviews (
    review_id bigserial NOT NULL,
    product_id integer NOT NULL,
    reviewer_id text,
    summary text,
    review_text text,
    vote text,
    overall real,
    verified boolean,
    review_time text,
    style text,
    PRIMARY KEY (review_id)
);

CREATE TABLE amazon.qa (
    qa_id bigserial NOT NULL,
    product_id integer NOT NULL,
    question_type text,
    answer_type text,
    question text,
    answer text,
    answer_time text,
    PRIMARY KEY (qa_id)
);
