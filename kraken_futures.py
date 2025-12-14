#!/usr/bin/env python3
"""
Kraken-Futures API client.
1-to-1 translation of the official JS sample.
"""
import base64
import hashlib
import hmac
import time
import urllib.parse
from typing import Dict, Any, Optional

import requests


class KrakenFuturesApi:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "https://futures.kraken.com",
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self._nonce_counter = 0

    # ------------------------------------------------------------------
    # low-level helpers
    # ------------------------------------------------------------------
    def _create_nonce(self) -> str:
        if self._nonce_counter > 9_999:
            self._nonce_counter = 0
        counter_str = f"{self._nonce_counter:05d}"
        self._nonce_counter += 1
        return f"{int(time.time() * 1_000)}{counter_str}"

    def _sign_request(self, endpoint: str, nonce: str, post_data: str = "") -> str:
        # strip '/derivatives' prefix if present
        path = endpoint[12:] if endpoint.startswith("/derivatives") else endpoint
        message = (post_data + nonce + path).encode()
        sha256_hash = hashlib.sha256(message).digest()
        secret_decoded = base64.b64decode(self.api_secret)
        sig = hmac.new(secret_decoded, sha256_hash, hashlib.sha512).digest()
        return base64.b64encode(sig).decode()

    # ------------------------------------------------------------------
    # single universal request method
    # ------------------------------------------------------------------
    def _request(
        self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        params = params or {}
        url = self.base_url + endpoint
        nonce = self._create_nonce()
        post_data = ""
        headers = {
            "APIKey": self.api_key,
            "Nonce": nonce,
            "User-Agent": "Kraken-Futures-Py-Client/1.0",
        }

        if method.upper() == "POST":
            post_data = urllib.parse.urlencode(params)
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        elif params:
            url += "?" + urllib.parse.urlencode(params)

        headers["Authent"] = self._sign_request(endpoint, nonce, post_data)

        rsp = requests.request(method, url, headers=headers, data=post_data or None)
        if not rsp.ok:
            raise RuntimeError(f"{method} {endpoint} failed : {rsp.text}")
        return rsp.json()

    # ------------------------------------------------------------------
    # public endpoints
    # ------------------------------------------------------------------
    def get_instruments(self) -> Dict[str, Any]:
        return self._request("GET", "/derivatives/api/v3/instruments")

    def get_tickers(self) -> Dict[str, Any]:
        return self._request("GET", "/derivatives/api/v3/tickers")

    def get_orderbook(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("GET", "/derivatives/api/v3/orderbook", params)

    def get_history(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("GET", "/derivatives/api/v3/history", params)

    # ------------------------------------------------------------------
    # private endpoints
    # ------------------------------------------------------------------
    def get_accounts(self) -> Dict[str, Any]:
        return self._request("GET", "/derivatives/api/v3/accounts")

    def send_order(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/derivatives/api/v3/sendorder", params)

    def edit_order(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/derivatives/api/v3/editorder", params)

    def cancel_order(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/derivatives/api/v3/cancelorder", params)

    def cancel_all_orders(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("POST", "/derivatives/api/v3/cancelallorders", params)

    def cancel_all_orders_after(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/derivatives/api/v3/cancelallordersafter", params)

    def batch_order(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/derivatives/api/v3/batchorder", params)

    def get_open_orders(self) -> Dict[str, Any]:
        return self._request("GET", "/derivatives/api/v3/openorders")

    def get_open_positions(self) -> Dict[str, Any]:
        return self._request("GET", "/derivatives/api/v3/openpositions")

    def get_recent_orders(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("GET", "/derivatives/api/v3/recentorders", params)

    def get_fills(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("GET", "/derivatives/api/v3/fills", params)

    def get_account_log(self) -> Dict[str, Any]:
        return self._request("GET", "/api/history/v2/account-log")

    def get_transfers(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("GET", "/derivatives/api/v3/transfers", params)

    def get_notifications(self) -> Dict[str, Any]:
        return self._request("GET", "/derivatives/api/v3/notifications")

    def get_order(self, order_id: str) -> Dict[str, Any]:
        """Return single order status."""
        return self._request("GET", "/derivatives/api/v3/orders", {"order_id": order_id})

# ------------------------------------------------------------------
# quick self-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import os

    KEY = os.getenv("KRAKEN_FUTURES_KEY", "YOUR_API_KEY")
    SEC = os.getenv("KRAKEN_FUTURES_SECRET", "YOUR_API_SECRET")

    api = KrakenFuturesApi(KEY, SEC)

    print("--- public tickers ---")
    print(api.get_tickers()["tickers"][:2])

    print("\n--- private accounts ---")
    print(api.get_accounts())
