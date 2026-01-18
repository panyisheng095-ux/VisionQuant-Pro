"""
构建/扩充 K 线图像数据集（用于 FAISS 索引）

用法示例：
python3 scripts/build_kline_image_dataset.py --start-date 20100101 --end-date 20251231 --stride 8 --target-images 1000000
"""
import os
import sys
import time
import json
import math
import argparse
import multiprocessing as mp
from datetime import datetime
from typing import List, Optional, Dict, Tuple

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import mplfinance as mpf
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.data_loader import DataLoader


def _normalize_symbols(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty:
        return []
    for col in ["code", "symbol", "代码", "证券代码"]:
        if col in df.columns:
            return [str(x).strip().zfill(6) for x in df[col].dropna().tolist()]
    return []


def _load_symbols(loader: DataLoader, symbols_file: Optional[str], max_stocks: Optional[int]) -> List[str]:
    if symbols_file and os.path.exists(symbols_file):
        with open(symbols_file, "r", encoding="utf-8") as f:
            symbols = [line.strip().zfill(6) for line in f if line.strip()]
        return symbols[:max_stocks] if max_stocks else symbols

    try:
        df = loader.data_source.get_stock_list()
        symbols = _normalize_symbols(df)
        if not symbols:
            raise ValueError("股票列表为空")
        return symbols[:max_stocks] if max_stocks else symbols
    except Exception:
        # 回退到 AkShare
        try:
            import akshare as ak
            df = ak.stock_info_a_code_name()
            symbols = _normalize_symbols(df)
            return symbols[:max_stocks] if max_stocks else symbols
        except Exception:
            return []


def _scan_existing_images(output_dir: str) -> Tuple[int, set, Dict[str, int]]:
    existing_count = 0
    symbols_covered = set()
    year_counts: Dict[str, int] = {}
    if not os.path.exists(output_dir):
        return existing_count, symbols_covered, year_counts

    for root, _, files in os.walk(output_dir):
        for f in files:
            if not f.endswith(".png"):
                continue
            existing_count += 1
            name = f[:-4]
            parts = name.split("_")
            if len(parts) >= 2:
                sym = parts[0].zfill(6)
                date_str = parts[1]
                symbols_covered.add(sym)
                if len(date_str) >= 4 and date_str[:4].isdigit():
                    year = date_str[:4]
                    year_counts[year] = year_counts.get(year, 0) + 1
    return existing_count, symbols_covered, year_counts


def _load_progress(progress_file: str) -> Optional[dict]:
    if not progress_file or not os.path.exists(progress_file):
        return None
    try:
        with open(progress_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_progress(progress_file: str, data: dict):
    if not progress_file:
        return
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _estimate_combo(num_stocks: int, start_date: str, end_date: str, window: int,
                    stride: int, target_images: int) -> dict:
    try:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        if end_dt < start_dt:
            start_dt, end_dt = end_dt, start_dt
        trading_days = len(pd.bdate_range(start_dt, end_dt))
    except Exception:
        trading_days = 0

    per_stock = max(0, (trading_days - window) // max(1, stride) + 1) if trading_days > window else 0
    total_est = per_stock * num_stocks

    # 推荐 stride
    rec_stride = None
    if num_stocks > 0 and trading_days > window:
        desired_per_stock = max(1.0, target_images / num_stocks)
        rec_stride = (trading_days - window) / max(1.0, desired_per_stock - 1.0)
        rec_stride = int(max(1, round(rec_stride)))

    # 推荐时间跨度（年）
    rec_years = None
    if num_stocks > 0:
        required_days = target_images / num_stocks * max(1, stride) + window
        rec_years = required_days / 252.0

    # 推荐股票数
    rec_stocks = None
    if per_stock > 0:
        rec_stocks = int(math.ceil(target_images / per_stock))

    return {
        "trading_days": trading_days,
        "per_stock": per_stock,
        "estimated_total": int(total_est),
        "recommended_stride": rec_stride,
        "recommended_years": rec_years,
        "recommended_stocks": rec_stocks,
    }


def _format_years(year_counts: Dict[str, int]) -> str:
    if not year_counts:
        return "-"
    years = sorted([int(y) for y in year_counts.keys() if str(y).isdigit()])
    if not years:
        return "-"
    return f"{years[0]}-{years[-1]}({len(years)})"


def _generate_images_for_symbol_seq(
    symbol: str,
    df: pd.DataFrame,
    output_dir: str,
    window: int,
    stride: int,
    target_images: int,
    stats: dict,
    max_images_per_symbol: Optional[int] = None,
) -> dict:
    generated = 0
    years = set()

    if df is None or df.empty:
        return {"symbol": symbol, "generated": 0, "years": []}

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            return {"symbol": symbol, "generated": 0, "years": []}

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if len(df) < window:
        return {"symbol": symbol, "generated": 0, "years": []}

    symbol_dir = os.path.join(output_dir, symbol)
    os.makedirs(symbol_dir, exist_ok=True)

    market_colors = mpf.make_marketcolors(up="red", down="green", inherit=True)
    style = mpf.make_mpf_style(marketcolors=market_colors, gridstyle="")

    for i in range(window - 1, len(df), stride):
        if stats["count"] >= target_images:
            break
        if max_images_per_symbol and generated >= max_images_per_symbol:
            break
        date_dt = df.index[i]
        date_str = date_dt.strftime("%Y%m%d")
        img_path = os.path.join(symbol_dir, f"{symbol}_{date_str}.png")
        if os.path.exists(img_path):
            continue

        window_df = df.iloc[i - window + 1 : i + 1]
        if len(window_df) < window:
            continue
        try:
            mpf.plot(
                window_df,
                type="candle",
                style=style,
                savefig=dict(fname=img_path, dpi=50),
                figsize=(3, 3),
                axisoff=True,
                volume=False,
            )
            stats["count"] += 1
            stats["generated"] += 1
            generated += 1
            years.add(str(date_dt.year))
        except Exception:
            continue

    return {"symbol": symbol, "generated": generated, "years": list(years)}


def _worker_generate_images(args) -> dict:
    (symbol, data_source, start_date, end_date, adjust, output_dir, window, stride,
     target_images, max_images_per_symbol, jq_user, jq_pass, rq_user, rq_pass, shared_counter, lock) = args

    loader = DataLoader(
        data_source=data_source,
        jq_username=jq_user,
        jq_password=jq_pass,
        rq_username=rq_user,
        rq_password=rq_pass,
    )
    df = loader.get_stock_data(symbol, start_date=start_date, end_date=end_date, adjust=adjust, use_cache=True)

    generated = 0
    years = set()

    if df is None or df.empty:
        return {"symbol": symbol, "generated": 0, "years": []}

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            return {"symbol": symbol, "generated": 0, "years": []}

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if len(df) < window:
        return {"symbol": symbol, "generated": 0, "years": []}

    symbol_dir = os.path.join(output_dir, symbol)
    os.makedirs(symbol_dir, exist_ok=True)

    market_colors = mpf.make_marketcolors(up="red", down="green", inherit=True)
    style = mpf.make_mpf_style(marketcolors=market_colors, gridstyle="")

    for i in range(window - 1, len(df), stride):
        if shared_counter.value >= target_images:
            break
        if max_images_per_symbol and generated >= max_images_per_symbol:
            break
        date_dt = df.index[i]
        date_str = date_dt.strftime("%Y%m%d")
        img_path = os.path.join(symbol_dir, f"{symbol}_{date_str}.png")
        if os.path.exists(img_path):
            continue

        window_df = df.iloc[i - window + 1 : i + 1]
        if len(window_df) < window:
            continue
        try:
            mpf.plot(
                window_df,
                type="candle",
                style=style,
                savefig=dict(fname=img_path, dpi=50),
                figsize=(3, 3),
                axisoff=True,
                volume=False,
            )
            with lock:
                if shared_counter.value < target_images:
                    shared_counter.value += 1
                    generated += 1
                    years.add(str(date_dt.year))
                else:
                    break
        except Exception:
            continue

    return {"symbol": symbol, "generated": generated, "years": list(years)}


def _write_report(report_file: str, data: dict):
    if not report_file:
        return
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="构建/扩充K线图像数据集")
    parser.add_argument("--data-source", type=str, default="akshare", choices=["akshare", "jqdata", "rqdata"])
    parser.add_argument("--start-date", type=str, default="20100101")
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--auto-stride", action="store_true", default=False)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--target-images", type=int, default=1000000)
    parser.add_argument("--max-stocks", type=int, default=None)
    parser.add_argument("--symbols-file", type=str, default=None)
    parser.add_argument("--sleep", type=float, default=0.02)
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "data", "images"))
    parser.add_argument("--progress-file", type=str, default=os.path.join(PROJECT_ROOT, "data", "progress", "kline_image_build.json"))
    parser.add_argument("--report-file", type=str, default=os.path.join(PROJECT_ROOT, "data", "progress", "kline_image_report.json"))
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_true", default=False)
    parser.add_argument("--reset-progress", action="store_true", default=False)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--estimate-only", action="store_true", default=False)
    parser.add_argument("--adjust", type=str, default="qfq")
    parser.add_argument("--jq-user", type=str, default=None)
    parser.add_argument("--jq-pass", type=str, default=None)
    parser.add_argument("--rq-user", type=str, default=None)
    parser.add_argument("--rq-pass", type=str, default=None)
    parser.add_argument("--max-images-per-symbol", type=int, default=None)
    parser.add_argument("--skip-existing-scan", action="store_true", default=False)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-total", type=int, default=1)
    args = parser.parse_args()

    loader = DataLoader(
        data_source=args.data_source,
        jq_username=args.jq_user,
        jq_password=args.jq_pass,
        rq_username=args.rq_user,
        rq_password=args.rq_pass,
    )
    symbols = _load_symbols(loader, args.symbols_file, args.max_stocks)
    if not symbols:
        print("❌ 无法获取股票列表，请检查数据源或 symbols-file")
        sys.exit(1)

    # 分片（多机/多进程协作）
    if args.shard_total and args.shard_total > 1:
        if args.shard_index < 0 or args.shard_index >= args.shard_total:
            print("❌ shard-index 必须在 [0, shard-total) 范围内")
            sys.exit(1)
        symbols = [s for i, s in enumerate(symbols) if i % args.shard_total == args.shard_index]
        print(f"🔀 分片模式: shard {args.shard_index}/{args.shard_total} | 当前股票数: {len(symbols)}")

    end_date = args.end_date or datetime.now().strftime("%Y%m%d")

    # 估算建议
    estimate = _estimate_combo(len(symbols), args.start_date, end_date, args.window, args.stride, args.target_images)
    print("\n=== 估算建议 ===")
    print(f"股票数: {len(symbols)} | 交易日: {estimate['trading_days']} | 单股可生成: {estimate['per_stock']}")
    print(f"预计总图片: {estimate['estimated_total']:,}")
    if estimate["recommended_stride"]:
        print(f"建议 stride: {estimate['recommended_stride']} (以更接近目标 {args.target_images:,})")
    if estimate["recommended_years"]:
        print(f"若固定 stride={args.stride}，目标 {args.target_images:,} 需时间跨度≈{estimate['recommended_years']:.1f} 年")
    if estimate["recommended_stocks"]:
        print(f"若固定时间跨度，目标 {args.target_images:,} 需股票数≈{estimate['recommended_stocks']}")

    if args.auto_stride and estimate["recommended_stride"]:
        args.stride = max(1, estimate["recommended_stride"])
        print(f"✅ auto-stride 已应用: stride={args.stride}")

    # 候选 stride 组合估算
    stride_candidates = [6, 8, 10, 12]
    cand_lines = []
    for s in stride_candidates:
        est = _estimate_combo(len(symbols), args.start_date, end_date, args.window, s, args.target_images)
        cand_lines.append(f"stride={s} -> 预计总图 {est['estimated_total']:,}")
    print("候选组合：" + " | ".join(cand_lines))

    if args.estimate_only:
        print("仅估算完成（estimate-only），未生成图片。")
        return

    # 进度文件处理
    progress = None
    resume = args.resume and not args.no_resume
    if args.reset_progress:
        progress = None
    elif resume:
        progress = _load_progress(args.progress_file)

    processed_symbols = set(progress.get("processed_symbols", [])) if progress else set()

    # 扫描已有图片（用于断点续跑）
    if args.skip_existing_scan and progress:
        existing_count = progress.get("total_count", 0)
        covered_symbols = set(progress.get("covered_symbols", []))
        year_counts = progress.get("year_counts", {})
    else:
        existing_count, covered_symbols, year_counts = _scan_existing_images(args.output_dir)

    # 断点续跑：已覆盖股票加入 processed
    if resume:
        processed_symbols |= covered_symbols

    stats = {
        "count": existing_count,
        "generated": 0,
        "existing": existing_count
    }

    print("\n=== 断点续跑状态 ===")
    print(f"已有图片: {existing_count:,} | 已覆盖股票: {len(covered_symbols)} | 年份覆盖: {_format_years(year_counts)}")
    if stats["count"] >= args.target_images:
        print("已达到目标图片数，无需生成。")
        return

    pending_symbols = [s for s in symbols if s not in processed_symbols]

    print(f"\n📌 数据源: {args.data_source} | 股票数: {len(symbols)} | 待处理: {len(pending_symbols)}")
    print(f"📅 区间: {args.start_date} ~ {end_date} | stride={args.stride} | window={args.window}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"🧾 进度文件: {args.progress_file}")
    print(f"⚡ 多进程: {args.num_workers} (K线图渲染为CPU；索引重建支持GPU)")

    processed_count = 0
    pbar = tqdm(pending_symbols, desc="生成K线图")

    if args.num_workers <= 1:
        for symbol in pbar:
            if stats["count"] >= args.target_images:
                break
            df = loader.get_stock_data(symbol, start_date=args.start_date, end_date=end_date, adjust=args.adjust, use_cache=True)
            result = _generate_images_for_symbol_seq(
                symbol=symbol,
                df=df,
                output_dir=args.output_dir,
                window=args.window,
                stride=args.stride,
                target_images=args.target_images,
                stats=stats,
                max_images_per_symbol=args.max_images_per_symbol,
            )

            if result.get("generated", 0) > 0:
                covered_symbols.add(symbol)
                for y in result.get("years", []):
                    year_counts[y] = year_counts.get(y, 0) + 1

            processed_symbols.add(symbol)
            processed_count += 1

            pbar.set_postfix({
                "imgs": f"{stats['count']:,}",
                "covered": f"{len(covered_symbols)}/{len(symbols)}",
                "years": _format_years(year_counts)
            })

            if processed_count % args.checkpoint_interval == 0:
                progress = {
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "start_date": args.start_date,
                    "end_date": end_date,
                    "stride": args.stride,
                    "window": args.window,
                    "target_images": args.target_images,
                    "output_dir": args.output_dir,
                    "total_symbols": len(symbols),
                    "processed_symbols": sorted(list(processed_symbols)),
                    "covered_symbols": sorted(list(covered_symbols)),
                    "existing_count": stats["existing"],
                    "generated_count": stats["generated"],
                    "total_count": stats["count"],
                    "year_counts": year_counts,
                }
                _save_progress(args.progress_file, progress)

            time.sleep(args.sleep)
    else:
        manager = mp.Manager()
        shared_counter = manager.Value('i', stats["count"])
        lock = manager.Lock()

        pool = mp.Pool(processes=args.num_workers)
        try:
            tasks = [(
                s, args.data_source, args.start_date, end_date, args.adjust, args.output_dir,
                args.window, args.stride, args.target_images, args.max_images_per_symbol,
                args.jq_user, args.jq_pass, args.rq_user, args.rq_pass,
                shared_counter, lock
            ) for s in pending_symbols]

            for result in pool.imap_unordered(_worker_generate_images, tasks):
                if result.get("generated", 0) > 0:
                    covered_symbols.add(result.get("symbol"))
                    for y in result.get("years", []):
                        year_counts[y] = year_counts.get(y, 0) + 1
                    stats["generated"] += result.get("generated", 0)
                stats["count"] = shared_counter.value

                processed_symbols.add(result.get("symbol"))
                processed_count += 1

                pbar.update(1)
                pbar.set_postfix({
                    "imgs": f"{stats['count']:,}",
                    "covered": f"{len(covered_symbols)}/{len(symbols)}",
                    "years": _format_years(year_counts)
                })

                if processed_count % args.checkpoint_interval == 0:
                    progress = {
                        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "start_date": args.start_date,
                        "end_date": end_date,
                        "stride": args.stride,
                        "window": args.window,
                        "target_images": args.target_images,
                        "output_dir": args.output_dir,
                        "total_symbols": len(symbols),
                        "processed_symbols": sorted(list(processed_symbols)),
                        "covered_symbols": sorted(list(covered_symbols)),
                        "existing_count": stats["existing"],
                        "generated_count": stats["generated"],
                        "total_count": stats["count"],
                        "year_counts": year_counts,
                    }
                    _save_progress(args.progress_file, progress)

                if stats["count"] >= args.target_images:
                    break
        finally:
            pool.terminate()
            pool.join()

    # 最后保存一次进度 + 报告
    progress = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "start_date": args.start_date,
        "end_date": end_date,
        "stride": args.stride,
        "window": args.window,
        "target_images": args.target_images,
        "output_dir": args.output_dir,
        "total_symbols": len(symbols),
        "processed_symbols": sorted(list(processed_symbols)),
        "covered_symbols": sorted(list(covered_symbols)),
        "existing_count": stats["existing"],
        "generated_count": stats["generated"],
        "total_count": stats["count"],
        "year_counts": year_counts,
    }
    _save_progress(args.progress_file, progress)

    report = {
        "generated": stats["generated"],
        "total": stats["count"],
        "covered_symbols": len(covered_symbols),
        "total_symbols": len(symbols),
        "year_span": _format_years(year_counts),
        "year_counts": year_counts,
        "output_dir": args.output_dir,
    }
    _write_report(args.report_file, report)

    print(f"\n✅ 已生成图片数: {stats['generated']:,} | 总数: {stats['count']:,}")
    print(f"覆盖股票: {len(covered_symbols)}/{len(symbols)} | 年份覆盖: {_format_years(year_counts)}")
    print(f"覆盖报告: {args.report_file}")
    print("下一步: 运行 scripts/rebuild_index_attention.py 重建 FAISS 索引")


if __name__ == "__main__":
    main()
