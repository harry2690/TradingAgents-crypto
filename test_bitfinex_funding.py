import os
import time
import hmac
import json
import hashlib
import requests
import pytest

BASE_URL = "https://api.bitfinex.com"
API_KEY = os.environ["BITFINEX_API_KEY"]
API_SECRET = os.environ["BITFINEX_API_SECRET"].encode()

def _send_private(path, body=None):
    body = body or {}
    raw_body = json.dumps(body)
    nonce = str(int(time.time() * 1000))
    signature_payload = f"/api{path}{nonce}{raw_body}".encode()
    signature = hmac.new(API_SECRET, signature_payload, hashlib.sha384).hexdigest()

    headers = {
        "bfx-nonce": nonce,
        "bfx-apikey": API_KEY,
        "bfx-signature": signature,
        "content-type": "application/json",
    }
    url = f"{BASE_URL}{path}"
    resp = requests.post(url, headers=headers, data=raw_body, timeout=30)
    resp.raise_for_status()
    return resp.json()

@pytest.mark.order(1)
def test_get_wallets():
    data = _send_private("/v2/auth/r/wallets")
    assert isinstance(data, list), "API 未成功回傳錢包資料"

@pytest.mark.order(2)
def test_get_funding_credits():
    data = _send_private("/v2/auth/r/funding/credits")
    assert isinstance(data, list), "API 未成功回傳放貸資料"

@pytest.mark.order(3)
def test_submit_and_cancel_funding_offer():
    annual_rate = 1.0  # 年利率 100%
    daily_rate = annual_rate / 365  # 換算日利率
    offer_body = {
        "type": "LIMIT",
        "symbol": "fUSD",
        "amount": "151",
        "rate": f"{daily_rate:.10f}",
        "period": 2
    }
    offer_res = _send_private("/v2/auth/w/funding/offer/submit", offer_body)
    offer_id = offer_res[4]  # 回傳陣列中索引 4 為訂單 ID
    assert offer_id, "建立放貸訂單失敗"

    offers = _send_private("/v2/auth/r/offers/fUSD")
    assert any(o[0] == offer_id for o in offers), "查無剛建立的放貸訂單"

    cancel_res = _send_private("/v2/auth/w/funding/offer/cancel", {"id": offer_id})
    assert cancel_res[4] == offer_id, "取消訂單時發生錯誤"
