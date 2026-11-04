from __future__ import annotations

from pathlib import Path
import json
import re
import pandas as pd


class DataValidationError(ValueError):
    """Raised when a required dataset is missing or structurally invalid."""


REQUIRED_COLUMNS = {
    "products": {"product_id", "price_brl", "name", "category_id", "description", "stock_quantity", "status", "specs"},
    "orders": {"order_id", "customer_id", "order_date", "status", "total_brl", "payment_method", "tracking_code", "estimated_delivery", "notes"},
    "order_items": {"order_id", "quantity", "product_id"},
    "customers": {"customer_id", "name", "phone", "email", "city"},
    "categories": {"category_id", "name", "description"},
    "promotions": {"promotion_id", "product_id", "discount_percent", "description", "is_active"},
}


def _clean_text(value):
    if pd.isna(value):
        return value
    value = re.sub(r"\s+", " ", str(value)).strip()
    # Repair common UTF-8 decoded as Latin-1/Windows-1252 corruption when detectable.
    if any(token in value for token in ("Ã", "Â", "�")):
        try:
            repaired = value.encode("latin1").decode("utf-8")
            if repaired:
                value = repaired
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataValidationError(f"Required dataset not found: {path}")
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp1252")
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].map(_clean_text)
    return df


class StoreData:
    """Loads the store's CSV datasets and validates their basic structure."""

    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)
        self.products = _read_csv(self.data_dir / "products.csv")
        self.orders = _read_csv(self.data_dir / "orders.csv")
        self.order_items = _read_csv(self.data_dir / "order_items.csv")
        self.customers = _read_csv(self.data_dir / "customers.csv")
        self.categories = _read_csv(self.data_dir / "categories.csv")
        self.promotions = _read_csv(self.data_dir / "promotions.csv")
        self._validate()

    def _validate(self):
        for name in REQUIRED_COLUMNS:
            frame = getattr(self, name)
            missing = REQUIRED_COLUMNS[name] - set(frame.columns)
            if missing:
                raise DataValidationError(f"{name}.csv is missing columns: {sorted(missing)}")

        # IDs must be unique in entity tables.
        for name, key in [("products", "product_id"), ("orders", "order_id"),
                          ("customers", "customer_id"), ("categories", "category_id"),
                          ("promotions", "promotion_id")]:
            frame = getattr(self, name)
            if frame[key].duplicated().any():
                raise DataValidationError(f"Duplicate {key} values found in {name}.csv")

        # Referential integrity checks are reported, not silently repaired.
        product_ids = set(self.products.product_id.astype(int))
        order_ids = set(self.orders.order_id.astype(int))
        customer_ids = set(self.customers.customer_id.astype(int))
        category_ids = set(self.categories.category_id.astype(int))
        if not set(self.order_items.product_id.astype(int)).issubset(product_ids):
            raise DataValidationError("order_items.csv contains unknown product_id values")
        if not set(self.order_items.order_id.astype(int)).issubset(order_ids):
            raise DataValidationError("order_items.csv contains unknown order_id values")
        if not set(self.orders.customer_id.astype(int)).issubset(customer_ids):
            raise DataValidationError("orders.csv contains unknown customer_id values")
        if not set(self.products.category_id.astype(int)).issubset(category_ids):
            raise DataValidationError("products.csv contains unknown category_id values")
        if not set(self.promotions.product_id.astype(int)).issubset(product_ids):
            raise DataValidationError("promotions.csv contains unknown product_id values")

    def diagnostics(self) -> dict:
        """Return a compact, non-sensitive dataset health report."""
        item_totals = (
            self.order_items.merge(self.products[["product_id", "price_brl"]], on="product_id")
            .assign(line_total=lambda x: x.quantity * x.price_brl)
            .groupby("order_id").line_total.sum()
        )
        order_totals = self.orders.set_index("order_id").total_brl
        mismatches = []
        for order_id, calculated in item_totals.items():
            recorded = float(order_totals.loc[order_id])
            if abs(recorded - float(calculated)) > 0.01:
                mismatches.append({
                    "order_id": int(order_id),
                    "recorded_total_brl": round(recorded, 2),
                    "sum_of_items_brl": round(float(calculated), 2),
                    "difference_brl": round(recorded - float(calculated), 2),
                })
        return {
            "counts": {
                "products": len(self.products), "orders": len(self.orders),
                "order_items": len(self.order_items), "customers": len(self.customers),
                "categories": len(self.categories), "promotions": len(self.promotions),
            },
            "product_status": self.products.status.value_counts().to_dict(),
            "order_status": self.orders.status.value_counts().to_dict(),
            "active_promotions": int((self.promotions.is_active.astype(int) == 1).sum()),
            "order_total_mismatches": mismatches,
        }
