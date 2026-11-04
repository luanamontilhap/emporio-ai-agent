from __future__ import annotations

from datetime import date
import re
import pandas as pd

from .data_loader import StoreData


def _brl(value: float) -> str:
    return ("R$ " + f"{value:,.2f}").replace(",", "X").replace(".", ",").replace("X", ".")


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


class StoreTools:
    """Deterministic business tools. These functions do not call an LLM."""

    def __init__(self, data: StoreData):
        self.data = data

    def search_products(
        self,
        query: str | None = None,
        category: str | None = None,
        max_price_brl: float | None = None,
        available_only: bool = True,
        limit: int = 10,
    ) -> dict:
        """Search catalog by text/category/price and return current dataset stock."""
        if limit < 1 or limit > 50:
            return {"ok": False, "error": "limit must be between 1 and 50."}
        df = self.data.products.merge(
            self.data.categories[["category_id", "name"]].rename(columns={"name": "category_name"}),
            on="category_id", how="left"
        )
        if query:
            q = _normalize(query)
            mask = (
                df.name.fillna("").map(_normalize).str.contains(re.escape(q), regex=True)
                | df.description.fillna("").map(_normalize).str.contains(re.escape(q), regex=True)
                | df.specs.fillna("").map(_normalize).str.contains(re.escape(q), regex=True)
            )
            df = df[mask]
        if category:
            c = _normalize(category)
            df = df[df.category_name.fillna("").map(_normalize).str.contains(re.escape(c), regex=True)]
        if max_price_brl is not None:
            if max_price_brl < 0:
                return {"ok": False, "error": "max_price_brl cannot be negative."}
            df = df[df.price_brl <= float(max_price_brl)]

        df = df.copy()
        df["stock_quantity"] = pd.to_numeric(df.stock_quantity, errors="coerce").fillna(0).astype(int)
        # Active promotions are read from the CSV flag. The dataset has no validity dates.
        active_promos = self.data.promotions[self.data.promotions.is_active.astype(int) == 1]
        promo_by_product = (
            active_promos.sort_values("discount_percent", ascending=False)
            .drop_duplicates("product_id").set_index("product_id")
        )
        results = []
        for row in df.itertuples(index=False):
            status = str(row.status).casefold()
            available = status == "active" and row.stock_quantity > 0
            if available_only and not available:
                continue
            promo = promo_by_product.loc[row.product_id] if row.product_id in promo_by_product.index else None
            discount = int(promo.discount_percent) if promo is not None else 0
            original_price = float(row.price_brl)
            promotional_price = round(original_price * (1 - discount / 100), 2) if discount else None
            results.append({
                "product_id": int(row.product_id),
                "name": row.name,
                "category": row.category_name,
                "description": row.description,
                "price_brl": round(original_price, 2),
                "price_formatted": _brl(original_price),
                "status": status,
                "stock_quantity": int(row.stock_quantity),
                "available_for_sale": available,
                "promotion": (
                    {"description": promo.description, "discount_percent": discount,
                     "promotional_price_brl": promotional_price,
                     "promotional_price_formatted": _brl(promotional_price)}
                    if promo is not None else None
                ),
            })
        results.sort(key=lambda x: (x["price_brl"], x["name"]))
        return {"ok": True, "count": min(len(results), limit), "total_matches": len(results),
                "products": results[:limit], "source": "products.csv + promotions.csv"}

    def get_product_details(self, product_id: int) -> dict:
        """Get a product by its exact identifier, including stock and active promotion."""
        matches = self.data.products[self.data.products.product_id == int(product_id)]
        if matches.empty:
            return {"ok": False, "error": f"Product ID {product_id} was not found."}
        # Reuse the common formatter/pricing logic without imposing the search result limit.
        product_row = matches.iloc[0]
        category_rows = self.data.categories[self.data.categories.category_id == int(product_row.category_id)]
        category_name = category_rows.iloc[0]["name"] if not category_rows.empty else None
        active_promos = self.data.promotions[
            (self.data.promotions.product_id == int(product_id))
            & (self.data.promotions.is_active.astype(int) == 1)
        ].sort_values("discount_percent", ascending=False)
        promo = None
        if not active_promos.empty:
            promo_row = active_promos.iloc[0]
            discount = int(promo_row.discount_percent)
            promo_price = round(float(product_row.price_brl) * (1 - discount / 100), 2)
            promo = {"description": promo_row.description, "discount_percent": discount,
                     "promotional_price_brl": promo_price,
                     "promotional_price_formatted": _brl(promo_price)}
        status = str(product_row.status).casefold()
        stock = int(product_row.stock_quantity)
        product = {
            "product_id": int(product_row.product_id), "name": product_row["name"],
            "category": category_name, "description": product_row.description,
            "price_brl": round(float(product_row.price_brl), 2),
            "price_formatted": _brl(float(product_row.price_brl)),
            "status": status, "stock_quantity": stock,
            "available_for_sale": status == "active" and stock > 0,
            "promotion": promo,
        }
        return {"ok": True, "product": product}

    def get_product_price(self, product_id: int, payment_method: str = "standard") -> dict:
        """Return list/promotional price and apply PIX discount only when no promotion exists."""
        result = self.get_product_details(product_id)
        if not result["ok"]:
            return result
        product = result["product"]
        base = product["price_brl"]
        promo = product["promotion"]
        if promo:
            final_price = promo["promotional_price_brl"]
            applied_rule = f"Active promotion: {promo['discount_percent']}%."
        elif payment_method.casefold() == "pix":
            final_price = round(base * 0.95, 2)
            applied_rule = "5% PIX discount applied to the list price."
        else:
            final_price = base
            applied_rule = "List price; no discount applied."
        return {
            "ok": True, "product_id": product["product_id"], "name": product["name"],
            "list_price_brl": base, "active_promotion": promo,
            "payment_method": payment_method, "final_price_brl": final_price,
            "final_price_formatted": _brl(final_price), "pricing_rule": applied_rule,
            "note": "PIX discount is not cumulative with promotional prices.",
        }

    def get_order_details(self, order_id: int, customer_phone: str) -> dict:
        """Retrieve an order only after matching its customer phone to the customer record.

        This is a prototype verification step, not production-grade authentication.
        """
        order_rows = self.data.orders[self.data.orders.order_id == int(order_id)]
        if order_rows.empty:
            return {"ok": False, "error": "Order not found."}
        order = order_rows.iloc[0]
        customer_rows = self.data.customers[
            self.data.customers.customer_id == int(order.customer_id)
        ]
        if customer_rows.empty:
            return {"ok": False, "error": "Customer record unavailable; contact store support."}
        stored_phone = re.sub(r"\D", "", str(customer_rows.iloc[0].phone))
        supplied_phone = re.sub(r"\D", "", str(customer_phone))
        if not supplied_phone or supplied_phone != stored_phone:
            return {"ok": False, "error": "We could not verify the order owner. Please check the phone number."}

        items = self.data.order_items[self.data.order_items.order_id == int(order_id)]
        item_details = []
        joined = items.merge(self.data.products[["product_id", "name", "price_brl"]], on="product_id", how="left")
        for item in joined.itertuples(index=False):
            item_details.append({
                "product_id": int(item.product_id), "product_name": item.name,
                "quantity": int(item.quantity), "unit_price_brl": round(float(item.price_brl), 2),
                "line_total_brl": round(float(item.quantity) * float(item.price_brl), 2),
            })
        return {
            "ok": True,
            "order": {
                "order_id": int(order.order_id), "status": order.status,
                "order_date": order.order_date, "total_brl": round(float(order.total_brl), 2),
                "payment_method": order.payment_method,
                "tracking_code": None if pd.isna(order.tracking_code) else order.tracking_code,
                "estimated_delivery": None if pd.isna(order.estimated_delivery) else order.estimated_delivery,
                "notes": None if pd.isna(order.notes) else order.notes,
                "items": item_details,
            },
            "privacy": "Customer identity was checked against the registered phone number.",
            "data_freshness": "The result reflects the supplied prototype CSV snapshot, not a live store system.",
        }

    def get_customer_orders(self, customer_phone: str) -> dict:
        """List a customer's orders after exact phone-number verification."""
        supplied_phone = re.sub(r"\D", "", str(customer_phone))
        matches = self.data.customers[
            self.data.customers.phone.astype(str).map(lambda x: re.sub(r"\D", "", x)) == supplied_phone
        ]
        if len(matches) != 1:
            return {"ok": False, "error": "Customer could not be uniquely verified."}
        customer_id = int(matches.iloc[0].customer_id)
        orders = self.data.orders[self.data.orders.customer_id == customer_id]
        return {
            "ok": True, "order_count": len(orders),
            "orders": orders[["order_id", "order_date", "status", "total_brl"]]
                .sort_values("order_date", ascending=False).to_dict(orient="records"),
            "privacy": "Only orders belonging to the verified phone number are returned.",
        }

    def get_store_information(self) -> dict:
        """Return general store information transcribed from the supplied policy manual."""
        return {
            "ok": True,
            "store": "Empório da Música",
            "address": "Rua 14 de Maio, 3200 — Centro, Campo Grande - MS, CEP 79202-333",
            "landline": "(67) 3341-4444",
            "email": "contato@empóriodamusica.com.br",
            "opening_hours": {
                "monday_to_friday": "09:00–18:00",
                "saturday": "09:00–13:00",
                "sunday_and_holidays": "Closed",
            },
            "scope": "Musical instruments only; accessories are not sold.",
            "source": "Store policy manual, version 2.1, June 2025.",
            "note": "The manual lists a separate WhatsApp number; the channel number should be confirmed before production use.",
        }
