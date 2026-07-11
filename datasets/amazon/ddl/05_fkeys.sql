-- Foreign keys (run AFTER data load)

ALTER TABLE amazon.products
    ADD CONSTRAINT fk_products_brand_id
    FOREIGN KEY (brand_id) REFERENCES amazon.brands (brand_id);

ALTER TABLE amazon.product_categories
    ADD CONSTRAINT fk_product_categories_product_id
    FOREIGN KEY (product_id) REFERENCES amazon.products (product_id);

ALTER TABLE amazon.product_categories
    ADD CONSTRAINT fk_product_categories_category_id
    FOREIGN KEY (category_id) REFERENCES amazon.categories (category_id);

ALTER TABLE amazon.product_colors
    ADD CONSTRAINT fk_product_colors_product_id
    FOREIGN KEY (product_id) REFERENCES amazon.products (product_id);

ALTER TABLE amazon.product_colors
    ADD CONSTRAINT fk_product_colors_color_id
    FOREIGN KEY (color_id) REFERENCES amazon.colors (color_id);

ALTER TABLE amazon.also_buy
    ADD CONSTRAINT fk_also_buy_product_id
    FOREIGN KEY (product_id) REFERENCES amazon.products (product_id);

ALTER TABLE amazon.also_buy
    ADD CONSTRAINT fk_also_buy_related_product_id
    FOREIGN KEY (related_product_id) REFERENCES amazon.products (product_id);

ALTER TABLE amazon.also_view
    ADD CONSTRAINT fk_also_view_product_id
    FOREIGN KEY (product_id) REFERENCES amazon.products (product_id);

ALTER TABLE amazon.also_view
    ADD CONSTRAINT fk_also_view_related_product_id
    FOREIGN KEY (related_product_id) REFERENCES amazon.products (product_id);

ALTER TABLE amazon.reviews
    ADD CONSTRAINT fk_reviews_product_id
    FOREIGN KEY (product_id) REFERENCES amazon.products (product_id);

ALTER TABLE amazon.qa
    ADD CONSTRAINT fk_qa_product_id
    FOREIGN KEY (product_id) REFERENCES amazon.products (product_id);
