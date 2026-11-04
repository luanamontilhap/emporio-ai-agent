# Data audit — supplied prototype snapshot

| Dataset | Rows | Notes |
|---|---:|---|
| products.csv | 65 | 62 active, 2 discontinued, 1 coming_soon |
| orders.csv | 20 | 7 delivered, 4 confirmed, 4 pending, 1 shipped, 4 cancelled |
| order_items.csv | 22 | All order and product references validated |
| customers.csv | 50 | IDs 1–50 |
| categories.csv | 9 | Product catalog references categories 1, 2, 3, 4, 5 and 9 |
| promotions.csv | 25 | 4 rows have is_active = 1; 21 have is_active = 0 |

## Financial discrepancies

| Order | Recorded total | Sum of listed items | Recorded minus item sum |
|---:|---:|---:|---:|
| 3 | R$ 3,450.00 | R$ 3,498.00 | -R$ 48.00 |
| 20 | R$ 1,400.00 | R$ 1,488.00 | -R$ 88.00 |

The source does not explain these differences. The implementation preserves the recorded order total and reports discrepancies separately.

## Important correction to an earlier preliminary note

The promotions CSV contains four active rows, not zero. The implementation and this audit use the actual `is_active` values from the uploaded file.
