import requests
import time
import json
import logging
import os
from datetime import datetime


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://api.coingecko.com/api/v3"
RATE_LIMIT_DELAY = 1.2

SUPPORTED_COINS = ["bitcoin", "ethereum", "solana"]

DEFAULT_HISTORY_DAYS = 180


def fetch_current_price(coin_id: str, currency: str = "usd",
                        max_retries: int = 3) -> dict | None:
   
    url = f"{BASE_URL}/simple/price"

    params = {
        "ids": coin_id,
        "vs_currencies": currency,
        "include_24hr_change": "true",
        "include_market_cap": "true",
        "include_24hr_vol": "true"
    }

    for attempt in range(1, max_retries+1):
        try:
            logger.info(f"Fetching price for {coin_id} (attempt {attempt}/{max_retries})")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if coin_id not in data:
                logger.warning(f"Coin '{coin_id}' is not found " +
                               "in API Response")
                return None
            
            coin_data = data[coin_id]

            raw_change = coin_data.get(f"{currency}_24h_change")

            result = {
                "coin_id":    coin_id,
                "price":      coin_data.get(currency),
                "change_24h": round(raw_change, 4) if raw_change is not None else None,
                "market_cap": coin_data.get(f"{currency}_market_cap"),
                "volume_24h": coin_data.get(f"{currency}_24h_vol"),   # ← this was the broken key
                "currency":   currency
            }

            logger.info(f'Successfully fetched {coin_id}: ' + 
                        f'${result['price']:,.2f}')
            return result
        
        except requests.exceptions.Timeout:
            logger.warning(f"Attempt {attempt}: " +
                           f"Request timed out for {coin_id}")
            if attempt < max_retries:
                wait = 2**max_retries
                logger.info(f"Waiting {wait}s before retry...")
                time.sleep(wait)
        
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else None

            if status == 429:
                logger.warning("Rate limited (429). Waiting 60s...")
                time.sleep(60)
            elif status == 404:
                logger.error(f"Coin '{coin_id}' does not exist (404)")
                return None
            else:
                logger.error(f"HTTP error {status}: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    
        except requests.exceptions.ConnectionError:
            logger.error("Connection error — check internet connection")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
        
        except Exception as e:
            logger.error(f"Unexpected error fetching {coin_id}: {e}")
            return None
    
    logger.error(f"All {max_retries} retries exhausted for {coin_id}")
    return None


def fetch_price_history(coin_id: str, days: int = 90, currency: str = "usd",
                        max_retries: int = 3) -> list | None:
    url = f"{BASE_URL}/coins/{coin_id}/market_chart"

    params = {
        "vs_currency": currency,
        "days": days,
        "interval": "daily"
    }

    for attempt in range(1, max_retries+1):
        try:           
            logger.info(
                f"Fetching {days}-day history for {coin_id} "
                f"(attempt {attempt}/{max_retries})"
            )
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            prices = data.get('prices', [])

            if not prices:
                logger.warning(f"Empty price history for {coin_id}")
                return None
            logger.info(f"Retrieved {len(prices)} data points for {coin_id}")
            return prices

        except requests.exceptions.Timeout:
            logger.warning(f"Attempt {attempt}: Timeout fetching " +
                           f"history for {coin_id}")
            if attempt < max_retries:
                wait = 2 ** attempt
                time.sleep(wait)

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else None
            if status == 429:
                logger.warning("Rate limited. Waiting 60s...")
                time.sleep(60)
            else:
                logger.error(f"HTTP error {status}: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)

        except requests.exceptions.ConnectionError:
            logger.error("Connection error")
            if attempt < max_retries:
                time.sleep(2 ** attempt)

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None

    logger.error(f"All retries exhausted for {coin_id} history")
    return None


def save_raw_response(coin_id: str, data: dict | list, data_type: str) -> str:

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'{coin_id}_{data_type}_{timestamp}.json'
    filepath = os.path.join("data", "raw", filename)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Raw data saved to {filepath}")
    return filepath


def fetch_multiple_coins(
     coin_ids: list[str], currency: str = "usd") -> list[dict]:
    results = []

    for coin_id in coin_ids:
        data = fetch_current_price(coin_id, currency)

        if data:
            results.append(data)
        else:
            logger.warning(f"Skipping {coin_id} — fetch failed")

        time.sleep(RATE_LIMIT_DELAY)
    logger.info(f"Fetched {len(results)}/{len(coin_ids)} coins successfully")
    return results


if __name__ == "__main__":
    print("\n=== Testing fetcher.py ===\n")

    # Test 1: Current price
    print("Test 1: Fetching current Bitcoin price...")
    btc = fetch_current_price("bitcoin")
    if btc:
        print(f"  Price:     ${btc['price']:,.2f}")
        print(f"  24h Change: {btc['change_24h']:.2f}%")
        print(f"  Market Cap: ${btc['market_cap']:,.0f}")
    else:
        print("  FAILED — check internet connection")

    time.sleep(RATE_LIMIT_DELAY)

    # Test 2: Price history
    print("\nTest 2: Fetching 90-day Bitcoin history...")
    history = fetch_price_history("bitcoin", days=90)
    if history:
        print(f"  Data points: {len(history)}")
        print(f"  First entry: {history[0]}")
        print(f"  Last entry:  {history[-1]}")
        save_raw_response("bitcoin", history, "history")
    else:
        print("  FAILED")
            
        time.sleep(RATE_LIMIT_DELAY)

    # Test 3: Multiple coins
    print("\nTest 3: Fetching multiple coins...")
    coins = fetch_multiple_coins(["bitcoin", "ethereum", "solana"])
    for c in coins:
        print(f"  {c['coin_id'].upper()}: ${c['price']:,.2f}")

    print("\n=== All tests complete ===\n")