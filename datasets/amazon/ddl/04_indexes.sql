-- Indexes on FK columns (before FK constraints)

CREATE INDEX idx_products_brand_id ON amazon.products (brand_id);
CREATE INDEX idx_product_categories_category_id ON amazon.product_categories (category_id);
CREATE INDEX idx_product_colors_color_id ON amazon.product_colors (color_id);
CREATE INDEX idx_also_buy_related_product_id ON amazon.also_buy (related_product_id);
CREATE INDEX idx_also_view_related_product_id ON amazon.also_view (related_product_id);
CREATE INDEX idx_reviews_product_id ON amazon.reviews (product_id);
CREATE INDEX idx_qa_product_id ON amazon.qa (product_id);
