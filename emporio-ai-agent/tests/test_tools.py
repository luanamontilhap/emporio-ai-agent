from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emporio_agent.data_loader import StoreData
from emporio_agent.tools import StoreTools


def make_tools():
    return StoreTools(StoreData(PROJECT_ROOT / "data"))


def test_search_only_returns_active_in_stock_by_default():
    result = make_tools().search_products()
    assert result["ok"]
    assert all(p["status"] == "active" and p["stock_quantity"] > 0 for p in result["products"])


def test_product_price_uses_active_promotion_without_pix_stacking():
    tools = make_tools()
    # Product 127 is marked with an active promotion in the supplied promotions CSV.
    result = tools.get_product_price(127, "pix")
    assert result["ok"]
    assert result["active_promotion"] is not None
    assert result["final_price_brl"] == result["active_promotion"]["promotional_price_brl"]


def test_order_lookup_rejects_wrong_phone():
    result = make_tools().get_order_details(1, "(67) 00000-0000")
    assert not result["ok"]


def test_order_lookup_accepts_matching_phone():
    # Order 1 belongs to customer 3; phone is taken from the supplied customers table.
    result = make_tools().get_order_details(1, "(67) 98432-1098")
    assert result["ok"]
    assert result["order"]["order_id"] == 1


def test_store_information_contains_hours():
    result = make_tools().get_store_information()
    assert result["opening_hours"]["saturday"] == "09:00–13:00"


def test_product_details_can_retrieve_ids_above_first_50():
    result = make_tools().get_product_details(136)
    assert result["ok"]
    assert result["product"]["status"] == "discontinued"


def test_diagnostics_reports_known_total_mismatches():
    report = StoreData(PROJECT_ROOT / "data").diagnostics()
    mismatch_ids = {item["order_id"] for item in report["order_total_mismatches"]}
    assert mismatch_ids == {3, 20}


# --- Regression tests found while integrating the LLM layer -----------------

import json


def test_product_details_return_name_and_category_not_row_index():
    product = make_tools().get_product_details(127)["product"]
    assert product["name"] == "Kalani KAL-700T Tenor Natural"
    assert product["category"] == "Ukuleles"


def test_price_result_is_strictly_json_serializable():
    # Tool outputs are sent to the LLM as JSON; numpy scalars must not leak.
    json.dumps(make_tools().get_product_price(127, "pix"))
    json.dumps(make_tools().get_product_details(127))


def test_order_lookup_accepts_any_phone_formatting():
    tools = make_tools()
    for phone in ("67984321098", "(67)98432-1098", "+55 67 98432-1098".replace("+55 ", "")):
        assert tools.get_order_details(1, phone)["ok"], phone


def test_customer_orders_accepts_unformatted_phone():
    result = make_tools().get_customer_orders("67984321098")
    assert result["ok"] and result["order_count"] >= 1
