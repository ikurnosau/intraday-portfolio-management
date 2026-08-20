from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import pandas as pd
import pickle 
import numpy as np
import logging
import os
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeout
import tempfile
import threading
import time

from config.constants import Constants
from config.settings import Settings, get_settings
from data.object_store import B2ObjectStore
from data.processed.data_processing_utils import convert_to_eastern


class _NumpyCoreRedirectingUnpickler(pickle.Unpickler):
    """Unpickler that maps obsolete ``numpy._core`` → ``numpy.core``."""

    def find_class(self, module, name):
        # Redirect *any* submodule that starts with the obsolete prefix
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)
    

class _RequestRateLimiter:
    """Smooth request starts across threads to stay below an RPM limit."""

    def __init__(self, requests_per_minute: int):
        if requests_per_minute <= 0:
            raise ValueError("quote_requests_per_minute must be positive")
        self._interval = 60.0 / requests_per_minute
        self._next_request_at = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            request_at = max(now, self._next_request_at)
            self._next_request_at = request_at + self._interval
        delay = request_at - now
        if delay > 0:
            time.sleep(delay)


class _QuotePullMetrics:
    """Collect and periodically log concurrent quote-request metrics."""

    def __init__(self, report_interval_seconds: float = 10.0):
        self._report_interval_seconds = report_interval_seconds
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._started_at = now
            self._last_report_at = now
            self._last_report_attempts = 0
            self._last_report_quotes = 0
            self._attempts = 0
            self._pages = 0
            self._quotes = 0
            self._active = 0
            self._connection_errors = 0
            self._status_counts: dict[int, int] = defaultdict(int)
            self._latencies: list[float] = []

    def response_hook(self, response, *args, **kwargs):
        report = None
        with self._lock:
            self._attempts += 1
            self._status_counts[response.status_code] += 1
            if 200 <= response.status_code < 300:
                self._pages += 1
            self._latencies.append(response.elapsed.total_seconds())
            report = self._report_if_due_locked(time.monotonic())
        if report is not None:
            logging.info(report)
        return response

    def record_quotes(self, count: int) -> None:
        with self._lock:
            self._quotes += count

    def record_connection_error(self) -> None:
        report = None
        with self._lock:
            self._connection_errors += 1
            report = self._report_if_due_locked(time.monotonic())
        if report is not None:
            logging.info(report)

    def worker_started(self) -> None:
        with self._lock:
            self._active += 1

    def worker_finished(self) -> None:
        with self._lock:
            self._active -= 1

    def log_summary(self) -> None:
        with self._lock:
            has_new_activity = (
                self._attempts != self._last_report_attempts
                or self._quotes != self._last_report_quotes
            )
            report = (
                self._build_report_locked(time.monotonic())
                if has_new_activity
                else None
            )
        if report is not None:
            logging.info(report)

    def _report_if_due_locked(self, now: float) -> str | None:
        if now - self._last_report_at < self._report_interval_seconds:
            return None
        return self._build_report_locked(now)

    def _build_report_locked(self, now: float) -> str:
        elapsed = max(now - self._last_report_at, 1e-9)
        attempts_delta = self._attempts - self._last_report_attempts
        quotes_delta = self._quotes - self._last_report_quotes
        rpm = attempts_delta * 60.0 / elapsed
        quotes_per_second = quotes_delta / elapsed
        if self._latencies:
            latency_p50, latency_p95 = np.percentile(
                self._latencies,
                [50, 95],
            )
        else:
            latency_p50 = latency_p95 = 0.0

        self._last_report_at = now
        self._last_report_attempts = self._attempts
        self._last_report_quotes = self._quotes
        self._latencies.clear()
        return (
            "Alpaca quote pull: pid=%s pages=%s rpm=%.0f quotes/s=%.0f "
            "active=%s latency_p50=%.2fs latency_p95=%.2fs "
            "429=%s 504=%s connection_errors=%s"
            % (
                os.getpid(),
                self._pages,
                rpm,
                quotes_per_second,
                self._active,
                latency_p50,
                latency_p95,
                self._status_counts[429],
                self._status_counts[504],
                self._connection_errors,
            )
        )


def _retrieve_exact_quote_batch(
    sessions: list[tuple],
    worker_count: int,
    requests_per_minute: int,
) -> list[tuple]:
    """Run one process-local pool of independent symbol sessions."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    retriever = AlpacaMarketsRetriever(
        use_quote_estimation=False,
        exact_quote_workers=worker_count,
        exact_quote_processes=1,
        quote_requests_per_minute=requests_per_minute,
    )
    return retriever._retrieve_exact_quote_sessions(sessions)


class AlpacaMarketsRetriever:
    FEED = 'sip'
    QUOTE_FIELDS = ("ap", "as", "bp", "bs")
    QUOTE_COLUMNS = ("ask_price", "ask_size", "bid_price", "bid_size")
    QUOTE_SEED_LOOKBACK = timedelta(days=7)

    def __init__(
        self,
        timeframe: TimeFrame = TimeFrame.Minute,
        use_quote_estimation: bool = True,
        exact_quote_workers: int = 16,
        exact_quote_processes: int = 1,
        quote_requests_per_minute: int = 9_000,
        object_store: B2ObjectStore | None = None,
        settings: Settings | None = None,
    ):
        if exact_quote_workers <= 0:
            raise ValueError("exact_quote_workers must be positive")
        if exact_quote_processes <= 0:
            raise ValueError("exact_quote_processes must be positive")
        self.timeframe = timeframe
        self.use_quote_estimation = use_quote_estimation
        self.exact_quote_workers = exact_quote_workers
        self.exact_quote_processes = exact_quote_processes
        self.quote_requests_per_minute = quote_requests_per_minute
        self._client: StockHistoricalDataClient | None = None
        self._exact_quote_clients = threading.local()
        self._quote_request_limiter = _RequestRateLimiter(
            quote_requests_per_minute
        )
        self._quote_pull_metrics = _QuotePullMetrics()
        self._object_store = object_store
        self._settings = settings

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = get_settings()
        return self._settings

    @property
    def object_store(self) -> B2ObjectStore:
        """Lazily initialize B2 so non-cached live requests do not need B2 credentials."""
        if self._object_store is None:
            self._object_store = B2ObjectStore.from_settings(self.settings.b2)
        return self._object_store

    @property
    def client(self) -> StockHistoricalDataClient:
        """Lazy client: B2 cache loads do not require Alpaca credentials."""
        if self._client is None:
            alpaca_settings = self.settings.alpaca
            self._client = StockHistoricalDataClient(
                alpaca_settings.api_key,
                alpaca_settings.api_secret,
            )
        return self._client

    def _get_exact_quote_client(self) -> StockHistoricalDataClient:
        """Return a worker-local client; requests.Session is not thread-safe."""
        client = getattr(self._exact_quote_clients, "client", None)
        if client is None:
            alpaca_settings = self.settings.alpaca
            client = StockHistoricalDataClient(
                alpaca_settings.api_key,
                alpaca_settings.api_secret,
                raw_data=True,
            )
            client._session.hooks.setdefault("response", []).append(
                self._quote_pull_metrics.response_hook
            )
            self._exact_quote_clients.client = client
        return client

    def build_file_name(self,
                        symbol_or_symbols: str | list[str],
                        start: datetime,
                        end: datetime): 
        quote_source = (
            None
            if self.use_quote_estimation
            else "exact-quotes-asof-with-timestamp"
        )
        return self._build_file_name(
            symbol_or_symbols,
            start,
            end,
            suffix=quote_source,
        )

    def _build_bars_file_name(
        self,
        symbol_or_symbols: str | list[str],
        start: datetime,
        end: datetime,
    ) -> str:
        return self._build_file_name(symbol_or_symbols, start, end)

    def _build_file_name(
        self,
        symbol_or_symbols: str | list[str],
        start: datetime,
        end: datetime,
        suffix: str | None = None,
    ) -> str:
        symbols = (
            symbol_or_symbols
            if isinstance(symbol_or_symbols, list)
            else [symbol_or_symbols]
        )
        suffix_part = f"_{suffix}" if suffix else ""
        return (
            f"{self.timeframe}_{start.date()}-{end.date()}_"
            f"{'+'.join(symbols)[:100]}{suffix_part}.pkl"
        )
    
    @staticmethod
    def build_object_key(storage_prefix: str, file_name: str) -> str:
        return f"{storage_prefix.strip('/')}/{file_name}"

    def cache_exists(self, storage_prefix: str, file_name: str) -> bool:
        return self.object_store.exists(
            self.build_object_key(storage_prefix, file_name)
        )

    def save_data(self, payload: object, storage_prefix: str, file_name: str):
        object_key = self.build_object_key(storage_prefix, file_name)
        with tempfile.SpooledTemporaryFile(
            max_size=64 * 1024 * 1024,
            mode="w+b",
        ) as output_file:
            pickle.dump(payload, output_file, protocol=pickle.HIGHEST_PROTOCOL)
            output_file.seek(0)
            self.object_store.upload_fileobj(object_key, output_file)

    def load_data(
        self,
        storage_prefix: str,
        file_name: str,
    ) -> object:
        object_key = self.build_object_key(storage_prefix, file_name)
        with tempfile.SpooledTemporaryFile(
            max_size=64 * 1024 * 1024,
            mode="w+b",
        ) as input_file:
            self.object_store.download_fileobj(
                object_key,
                input_file,
            )
            input_file.seek(0)
            return _NumpyCoreRedirectingUnpickler(input_file).load()

    def get_all_symbols(self) -> list[str]:
        alpaca_settings = self.settings.alpaca
        trading_client = TradingClient(
            alpaca_settings.api_key,
            alpaca_settings.api_secret,
            paper=alpaca_settings.paper,
        )
        search_params = GetAssetsRequest(asset_class=AssetClass.US_EQUITY)

        assets = trading_client.get_all_assets(search_params)
        assets = [asset for asset in assets if \
                  asset.status == AssetStatus.ACTIVE and
                  asset.easy_to_borrow and
                  asset.fractionable and
                  not asset.min_order_size and
                  asset.shortable and
                  asset.tradable]
        return [asset.symbol for asset in assets]

    def get_history_depth(self, symbol: str) -> datetime:
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=self.timeframe,
            start=datetime(1900, 1, 1),
            limit=1,
            feed=self.FEED
        )
        bars = self.client.get_stock_bars(request_params).data[symbol]
        return bars[0].timestamp

    def has_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> bool:
        """Return whether a symbol has any daily bar in the given interval."""
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            limit=1,
            feed=self.FEED,
        )
        bars = self.client.get_stock_bars(request_params).data
        return bool(bars.get(symbol))

    def _bars(self,
             symbol_or_symbols: str | list[str],
             start: datetime=datetime(2025, 5, 1),
             end: datetime=datetime(2025, 5, 2), 
             save_dir: str=Constants.Data.Retrieving.Alpaca.BARS_STORAGE_PREFIX) -> dict[str: pd.DataFrame]:
        logging.info(
            "Retrieving Alpaca bars from the Alpaca API for %s to %s.",
            start,
            end,
        )
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol_or_symbols,
            timeframe=self.timeframe,
            start=start,
            end=end,
            feed=self.FEED
        )
        bars = self.client.get_stock_bars(request_params).data
        response = {}
        for symbol, stock_data in bars.items():
            df = pd.DataFrame([data_item.__dict__ for data_item in stock_data]) \
                    .drop(columns=['symbol', 'trade_count', 'vwap']) \
                    .rename(columns={'timestamp': 'date'})
            df = convert_to_eastern(df, 'date')
            response[symbol] = df
        
        if save_dir:
            file_name = self._build_bars_file_name(
                symbol_or_symbols,
                start,
                end,
            )
            self.save_data(response, save_dir, file_name)

        return response
    
    def bars(self,
             symbol_or_symbols: str | list[str],
             start: datetime=datetime(2025, 5, 1),
             end: datetime=datetime(2025, 5, 2), 
             save_dir: str=Constants.Data.Retrieving.Alpaca.BARS_STORAGE_PREFIX) -> dict[str: pd.DataFrame]:
        
        if save_dir:
            file_name = self._build_bars_file_name(
                symbol_or_symbols,
                start,
                end,
            )
            if self.cache_exists(save_dir, file_name):
                logging.info(
                    "Downloading cached Alpaca bars from cloud storage: %s",
                    self.build_object_key(save_dir, file_name),
                )
                data = self.load_data(save_dir, file_name)
                return {symbol: convert_to_eastern(df, 'date') for symbol, df in data.items()}
            
        return self._bars(symbol_or_symbols, start, end, save_dir)

    def _quote_estimation(self, symbol: str, start: datetime, end: datetime) -> dict[str: float]:
        start = pd.to_datetime(start).tz_convert(Constants.Data.EASTERN_TZ)
        end = pd.to_datetime(end).tz_convert(Constants.Data.EASTERN_TZ)
        start = datetime.combine(start.date(), Constants.Data.REGULAR_TRADING_HOURS_START)
        start = start.replace(tzinfo=Constants.Data.EASTERN_TZ) + timedelta(hours=2)

        rng = pd.date_range(start=start,
                            end=end,
                            freq="30d",
                            tz=Constants.Data.EASTERN_TZ,
                            inclusive="left")

        quotes = []
        for date in rng:
            start_date = pd.to_datetime(date)
            end_date = start_date + timedelta(hours=1)
            retrieval_result = self.quotes(symbol, start_date, end_date, limit=1)
            if symbol in retrieval_result:
                quotes.append(retrieval_result[symbol][0])

        avg_spread = np.mean([(quote.ask_price - quote.bid_price) for quote in quotes])
        ask_price = np.mean([quote.ask_price for quote in quotes])

        mean_ask_size = np.mean([quote.ask_size for quote in quotes])
        mean_bid_size = np.mean([quote.bid_size for quote in quotes])

        return {
            'ask_price': ask_price,
            'ask_size': int(mean_ask_size if np.isfinite(mean_ask_size) else 0),
            'bid_price': ask_price - avg_spread,
            'bid_size': int(mean_bid_size if np.isfinite(mean_ask_size) else 0),
        }

    def _timeframe_duration(self) -> pd.Timedelta:
        duration_units = {
            TimeFrameUnit.Minute: "min",
            TimeFrameUnit.Hour: "h",
            TimeFrameUnit.Day: "d",
            TimeFrameUnit.Week: "w",
        }
        try:
            unit = duration_units[self.timeframe.unit]
        except KeyError as exc:
            raise ValueError(
                f"Exact quote retrieval does not support {self.timeframe.unit} bars"
            ) from exc
        return pd.Timedelta(self.timeframe.amount, unit=unit)

    @staticmethod
    def _quote_timestamps_ns(quotes: list[dict]) -> np.ndarray:
        return pd.DatetimeIndex(
            pd.to_datetime(
                [quote["t"] for quote in quotes],
                utc=True,
                format="ISO8601",
            )
        ).as_unit("ns").asi8

    def _request_quote_page(
        self,
        symbol: str,
        params: dict,
    ) -> tuple[dict, list[dict]]:
        self._quote_request_limiter.acquire()
        try:
            response = self._get_exact_quote_client().get(
                path="/stocks/quotes",
                data=params,
            )
        except (RequestsConnectionError, RequestsTimeout):
            self._quote_pull_metrics.record_connection_error()
            raise
        quotes = response.get("quotes", {}).get(symbol, [])
        self._quote_pull_metrics.record_quotes(len(quotes))
        return response, quotes

    def _exact_quotes_for_session(
        self,
        symbol: str,
        session_start: pd.Timestamp,
        bar_ends: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Return the latest quote strictly before every bar end."""
        self._quote_pull_metrics.worker_started()
        try:
            bar_end_ns = (
                pd.DatetimeIndex(bar_ends)
                .tz_convert("UTC")
                .as_unit("ns")
                .asi8
            )
            values = np.full(
                (len(bar_ends), len(self.QUOTE_FIELDS)),
                np.nan,
            )
            timestamps_ns = np.full(
                len(bar_ends),
                pd.NaT.value,
                dtype=np.int64,
            )

            seed_params = StockQuotesRequest(
                symbol_or_symbols=symbol,
                start=(
                    session_start - self.QUOTE_SEED_LOOKBACK
                ).to_pydatetime(),
                end=session_start.to_pydatetime(),
                limit=1,
                feed=self.FEED,
            ).to_request_fields()
            seed_params["sort"] = "desc"
            _, seed_quotes = self._request_quote_page(
                symbol,
                seed_params,
            )
            if seed_quotes:
                seed_quote = seed_quotes[0]
                seed_timestamp_ns = self._quote_timestamps_ns(
                    seed_quotes
                )[0]
                values[:] = [
                    seed_quote[field] for field in self.QUOTE_FIELDS
                ]
                timestamps_ns[:] = seed_timestamp_ns

            params = StockQuotesRequest(
                symbol_or_symbols=symbol,
                start=session_start.to_pydatetime(),
                end=bar_ends.max().to_pydatetime(),
                feed=self.FEED,
            ).to_request_fields()
            params.update(limit=10_000, sort="asc")
            page_token = None
            page_count = 1

            while True:
                params["page_token"] = page_token
                response, quotes = self._request_quote_page(
                    symbol,
                    params,
                )
                page_count += 1
                if quotes:
                    quote_ns = self._quote_timestamps_ns(quotes)
                    quote_positions = np.searchsorted(
                        quote_ns,
                        bar_end_ns,
                        side="left",
                    ) - 1
                    valid_bars = quote_positions >= 0
                    selected_quotes = [
                        quotes[position]
                        for position in quote_positions[valid_bars]
                    ]
                    values[valid_bars] = [
                        [
                            quote[field]
                            for field in self.QUOTE_FIELDS
                        ]
                        for quote in selected_quotes
                    ]
                    timestamps_ns[valid_bars] = quote_ns[
                        quote_positions[valid_bars]
                    ]

                page_token = response.get("next_page_token")
                if page_token is None:
                    return values, timestamps_ns, page_count
        finally:
            self._quote_pull_metrics.worker_finished()

    def _add_exact_quotes(
        self,
        symbol: str,
        bar_df: pd.DataFrame,
    ) -> pd.DataFrame:
        return self._add_exact_quotes_to_bars({symbol: bar_df})[symbol]

    def _retrieve_exact_quote_sessions(
        self,
        sessions: list[tuple],
    ) -> list[tuple]:
        self._quote_pull_metrics.reset()
        session_results = []
        max_workers = min(self.exact_quote_workers, len(sessions))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._exact_quotes_for_session,
                    symbol,
                    session_start,
                    bar_ends,
                ): (symbol, positions)
                for symbol, positions, session_start, bar_ends in sessions
            }
            for future in as_completed(futures):
                symbol, positions = futures[future]
                try:
                    quote_values, timestamps_ns, page_count = future.result()
                except Exception:
                    self._quote_pull_metrics.log_summary()
                    raise
                session_results.append(
                    (
                        symbol,
                        positions,
                        quote_values,
                        timestamps_ns,
                        page_count,
                    )
                )
        self._quote_pull_metrics.log_summary()
        return session_results

    def _add_exact_quotes_to_bars(
        self,
        bars: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        results = {
            symbol: bar_df.assign(
                **{
                    column: np.nan
                    for column in self.QUOTE_COLUMNS
                }
            )
            for symbol, bar_df in bars.items()
        }
        for result in results.values():
            result["quote_timestamp"] = pd.Series(
                pd.NaT,
                index=result.index,
                dtype=pd.DatetimeTZDtype(
                    unit="ns",
                    tz=Constants.Data.EASTERN_TZ,
                ),
            )
        quote_timestamp_ns = {
            symbol: np.full(len(bar_df), pd.NaT.value, dtype=np.int64)
            for symbol, bar_df in bars.items()
        }
        sessions = []
        session_counts = {}
        for symbol, bar_df in bars.items():
            if bar_df.empty:
                session_counts[symbol] = 0
                continue
            period_start = pd.to_datetime(bar_df["date"])
            period_end = period_start + self._timeframe_duration()
            session_dates = period_start.dt.date
            session_counts[symbol] = len(session_dates.unique())
            for session_date in session_dates.unique():
                positions = np.flatnonzero(session_dates == session_date)
                sessions.append(
                    (
                        symbol,
                        positions,
                        period_start.iloc[positions].min(),
                        period_end.iloc[positions],
                    )
                )

        if not sessions:
            return results

        page_counts = {symbol: 0 for symbol in bars}
        process_count = min(
            self.exact_quote_processes,
            self.exact_quote_workers,
            self.quote_requests_per_minute,
            len(sessions),
        )
        if process_count == 1:
            session_results = self._retrieve_exact_quote_sessions(sessions)
        else:
            batches = [
                sessions[index::process_count]
                for index in range(process_count)
            ]
            workers_per_process = max(
                1,
                self.exact_quote_workers // process_count,
            )
            extra_workers = self.exact_quote_workers % process_count
            rpm_per_process = max(
                1,
                self.quote_requests_per_minute // process_count,
            )
            extra_rpm = self.quote_requests_per_minute % process_count
            session_results = []
            with ProcessPoolExecutor(max_workers=process_count) as executor:
                futures = [
                    executor.submit(
                        _retrieve_exact_quote_batch,
                        batch,
                        workers_per_process + (index < extra_workers),
                        rpm_per_process + (index < extra_rpm),
                    )
                    for index, batch in enumerate(batches)
                ]
                for future in as_completed(futures):
                    session_results.extend(future.result())

        for (
            symbol,
            positions,
            quote_values,
            session_quote_timestamp_ns,
            page_count,
        ) in session_results:
            page_counts[symbol] += page_count
            results[symbol].iloc[
                positions,
                [
                    results[symbol].columns.get_loc(column)
                    for column in self.QUOTE_COLUMNS
                ],
            ] = quote_values
            quote_timestamp_ns[symbol][positions] = (
                session_quote_timestamp_ns
            )
        for symbol, result in results.items():
            result["quote_timestamp"] = pd.Series(
                pd.to_datetime(
                    quote_timestamp_ns[symbol],
                    utc=True,
                ).tz_convert(Constants.Data.EASTERN_TZ),
                index=result.index,
            )
            missing_count = (
                result[list(self.QUOTE_COLUMNS)]
                .isna()
                .all(axis=1)
                .sum()
            )
            if missing_count:
                logging.warning(
                    "No prior Alpaca quote found for %s of %s %s bars.",
                    missing_count,
                    len(result),
                    symbol,
                )
            logging.info(
                "Retrieved exact Alpaca quotes for %s in %s pages across %s sessions.",
                symbol,
                page_counts[symbol],
                session_counts[symbol],
            )
        return results

    def _bars_with_quotes(self,
             symbol_or_symbols: str | list[str],
             start: datetime=datetime(2025, 5, 1),
             end: datetime=datetime(2025, 5, 2), 
             save_dir: str=Constants.Data.Retrieving.Alpaca.BARS_WITH_QUOTES_STORAGE_PREFIX) -> dict[str: pd.DataFrame]:
        bars = self.bars(
            symbol_or_symbols,
            start,
            end,
            save_dir=Constants.Data.Retrieving.Alpaca.BARS_STORAGE_PREFIX,
        )
        if self.use_quote_estimation:
            logging.info(
                "Retrieving Alpaca quote estimates from the Alpaca API for %s to %s.",
                start,
                end,
            )
            quotes = {
                symbol: self._quote_estimation(symbol, start, end)
                for symbol in bars
            }
            for symbol, bar_df in bars.items():
                for column_name, value in quotes[symbol].items():
                    bar_df[column_name] = value
        else:
            logging.info(
                "Retrieving exact end-of-period Alpaca quotes for %s to %s.",
                start,
                end,
            )
            bars = self._add_exact_quotes_to_bars(bars)
        
        if save_dir:
            file_name = self.build_file_name(symbol_or_symbols, start, end)
            self.save_data(bars, save_dir, file_name)

        return bars


    def bars_with_quotes(self,
             symbol_or_symbols: str | list[str],
             start: datetime=datetime(2025, 5, 1),
             end: datetime=datetime(2025, 5, 2), 
             save_dir: str=Constants.Data.Retrieving.Alpaca.BARS_WITH_QUOTES_STORAGE_PREFIX) -> dict[str: pd.DataFrame]:
        
        if save_dir:
            file_name = self.build_file_name(symbol_or_symbols, start, end)
            if self.cache_exists(save_dir, file_name):
                logging.info(
                    "Downloading cached Alpaca bars with quotes from cloud storage: %s",
                    self.build_object_key(save_dir, file_name),
                )
                data = self.load_data(save_dir, file_name)
                return {symbol: convert_to_eastern(df, 'date') for symbol, df in data.items()}
            
        return self._bars_with_quotes(symbol_or_symbols, start, end, save_dir)
        
    def latest_bars(self, symbol_or_symbols, limit=100, pull_n_days=3):
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol_or_symbols,
            timeframe=self.timeframe,
            start=datetime.now(timezone.utc) - timedelta(days=pull_n_days),
            feed=self.FEED,
        )
        bars = self.client.get_stock_bars(request_params).data
        response = {}
        for symbol, stock_data in bars.items():
            df = pd.DataFrame([data_item.__dict__ for data_item in stock_data]).tail(limit).reset_index(drop=True) \
                    .drop(columns=['symbol', 'trade_count']) \
                    .rename(columns={'timestamp': 'date'})
            if len(df) < limit:
                logging.warning(f"Not enough bars for pulled for {symbol} to satisfy the limit of {limit}; pulled {pull_n_days} days and got {len(df)} bars.")
                continue

            df = convert_to_eastern(df, 'date')
            response[symbol] = df
        return response

    def quotes(self,
               symbol_or_symbols,
               start=datetime(2025, 5, 1),
               end=datetime(2025, 5, 2),
               limit=None
               ):
        request_params = StockQuotesRequest(
            symbol_or_symbols=symbol_or_symbols,
            start=start,
            end=end,
            feed=self.FEED,
            limit=limit
        )
        quotes = self.client.get_stock_quotes(request_params).data
        return quotes

    def latest_quote(self, symbol_or_symbols):
        request_params = StockLatestQuoteRequest(
            symbol_or_symbols=symbol_or_symbols,
            feed=self.FEED
        )
        quotes = self.client.get_stock_latest_quote(request_params)

        quotes = {
            symbol: {
                'bid_price': quote.bid_price, 
                'ask_price': quote.ask_price, 
                'bid_size': quote.bid_size, 
                'ask_size': quote.ask_size
            } 
            for symbol, quote in quotes.items()
        }

        return quotes

    # def latest_spread(self, symbol_or_symbols):
    #     quotes = self.latest_quote(symbol_or_symbols)
    #     return {symbol: (quote.ask_price - quote.bid_price) / quote.bid_price
    #             for symbol, quote in quotes.items()}