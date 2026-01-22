import akshare as ak
import pandas as pd
import numpy as np
import time


class FundamentalMiner:
    def __init__(self, spot_cache_ttl_sec: int = 300, spot_retry: int = 2):
        # 缓存全市场 spot（ak.stock_zh_a_spot_em 很重，且易波动；缓存能显著降低 N/A）
        self._spot_cache_df = None
        self._spot_cache_ts = 0.0
        self._spot_cache_ttl_sec = spot_cache_ttl_sec
        self._spot_retry = spot_retry
        self._industry_cache = {}
        self._peers_cache = {}
        self._industry_map = {}
        self._industry_cons_cache = {}
        self._industry_map_ts = 0.0
        self._industry_map_ttl_sec = 24 * 3600
        self._industry_board_names = []
        self._industry_board_ts = 0.0
        self._industry_board_ttl_sec = 24 * 3600
        self._industry_name_cache = {}

    def get_stock_fundamentals(self, symbol):
        """
        获取深度财务指标 (含成长性与安全性分析)
        """
        symbol = str(symbol).strip().zfill(6)
        print(f"🔍 [财务分析] 正在透视 {symbol}...")

        # 默认结果结构扩展
        result = {
            "symbol": symbol,
            # 默认不要用 symbol 当 name，否则 UI 会出现 “300286(300286)” 这种重复且掩盖抓取失败
            "name": "",
            "pe_ttm": 0.0, "pb": 0.0, "total_mv": 0.0,
            "roe": 0.0, "net_profit_margin": 0.0, "asset_turnover": 0.6, "leverage": 1.0,
            "debt_asset_ratio": 0.0,
            # === 新增指标 ===
            "gross_margin": 0.0,  # 毛利率
            "current_ratio": 0.0,  # 流动比率 (偿债能力)
            "rev_growth": 0.0,  # 营收增长率
            "profit_growth": 0.0,  # 净利增长率
            "report_date": "最新"
            ,
            # === 状态字段：用于UI层判断“是否成功抓取”，避免把0当真 ===
            "_ok": {"spot": False, "finance": False},
            "_err": []
        }

        try:
            # 1. 实时估值
            try:
                spot_df = self._get_spot_df_cached(result)
                if spot_df is not None and not spot_df.empty:
                    code_col = next((c for c in spot_df.columns if '代码' in c), None)
                    if code_col:
                        target = spot_df[spot_df[code_col] == symbol]
                        if not target.empty:
                            def _pick_col(cols, keywords, prefer=None):
                                prefer = prefer or []
                                # 优先匹配含 prefer 的列
                                for c in cols:
                                    if all(k in str(c) for k in prefer):
                                        return c
                                for c in cols:
                                    if any(k in str(c) for k in keywords):
                                        return c
                                return None

                            cols = list(target.columns)
                            pe_col = _pick_col(cols, ["市盈率", "PE", "pe", "TTM"], prefer=["市盈率", "TTM"])
                            pb_col = _pick_col(cols, ["市净率", "PB", "pb"])
                            mv_col = _pick_col(cols, ["总市值", "市值"])
                            name_col = next((c for c in target.columns if '名称' in c), None)

                            if pe_col:
                                result["pe_ttm"] = self._to_f(target[pe_col].values[0])
                            if pb_col:
                                result["pb"] = self._to_f(target[pb_col].values[0])
                            if mv_col:
                                result["total_mv"] = round(self._to_f(target[mv_col].values[0]) / 100000000, 2)
                            if name_col:
                                result["name"] = str(target[name_col].values[0]).strip()
                            result["_ok"]["spot"] = True
            except Exception as e:
                result["_err"].append(f"spot_df_error: {type(e).__name__}: {e}")

            # 若 spot 未拿到 name 或 PE/PB，尝试更轻量的个股信息接口兜底（带重试）
            if not result.get("name") or (result.get("pe_ttm", 0) == 0 and result.get("pb", 0) == 0):
                for attempt in range(max(1, self._spot_retry + 1)):
                    try:
                        info_df = ak.stock_individual_info_em(symbol=symbol)
                        if info_df is not None and not info_df.empty:
                            # 常见字段：item/value
                            if "item" in info_df.columns and "value" in info_df.columns:
                                # 获取股票名称
                                if not result.get("name"):
                                    name_row = info_df[info_df["item"].astype(str).str.contains("股票简称|名称")]
                                    if not name_row.empty:
                                        result["name"] = str(name_row["value"].values[0]).strip()
                                
                                # 尝试获取PE/PB（如果spot_df没有获取到）
                                if result.get("pe_ttm", 0) == 0:
                                    pe_row = info_df[info_df["item"].astype(str).str.contains("市盈率|PE")]
                                    if not pe_row.empty:
                                        result["pe_ttm"] = self._to_f(pe_row["value"].values[0])
                                
                                if result.get("pb", 0) == 0:
                                    pb_row = info_df[info_df["item"].astype(str).str.contains("市净率|PB")]
                                    if not pb_row.empty:
                                        result["pb"] = self._to_f(pb_row["value"].values[0])
                                
                                # 如果获取到关键数据，退出重试循环
                                if result.get("name") or result.get("pe_ttm", 0) > 0:
                                    break
                        if attempt < self._spot_retry:
                            time.sleep(0.5 * (attempt + 1))  # 指数退避
                    except Exception as e:
                        if attempt < self._spot_retry:
                            time.sleep(0.5 * (attempt + 1))
                            continue
                        result["_err"].append(f"stock_individual_info_error: {type(e).__name__}: {e}")

            # 1.5 估值指标兜底：从指标接口补齐 PE/PB/市值
            if (result.get("pe_ttm", 0) == 0 or result.get("pb", 0) == 0 or result.get("total_mv", 0) == 0):
                try:
                    extra = self._get_indicator_snapshot(symbol, max_retries=max(1, self._spot_retry))
                    if extra:
                        if result.get("pe_ttm", 0) == 0 and extra.get("pe_ttm") is not None:
                            result["pe_ttm"] = extra.get("pe_ttm", 0.0)
                        if result.get("pb", 0) == 0 and extra.get("pb") is not None:
                            result["pb"] = extra.get("pb", 0.0)
                        if result.get("total_mv", 0) == 0 and extra.get("total_mv") is not None:
                            result["total_mv"] = extra.get("total_mv", 0.0)
                        if extra.get("ok"):
                            result["_ok"]["spot"] = True
                except Exception as e:
                    result["_err"].append(f"indicator_fallback_error: {type(e).__name__}: {e}")

            # 2. 深度指标：优先使用 THS 财务摘要（经验证可用；EM接口在你环境里全量报错）
            # 工业级优化：添加重试机制
            ths_success = False
            for attempt in range(max(1, self._spot_retry + 1)):
                try:
                    ths_df = ak.stock_financial_abstract_ths(symbol=symbol)
                    if ths_df is not None and not ths_df.empty:
                        # 取最新报告期
                        if "报告期" in ths_df.columns:
                            tmp = ths_df.copy()
                            tmp["报告期_dt"] = pd.to_datetime(tmp["报告期"], errors="coerce")
                            tmp = tmp.sort_values("报告期_dt")
                            latest = tmp.iloc[-1]
                            result["report_date"] = str(latest.get("报告期", result["report_date"]))
                        else:
                            latest = ths_df.iloc[-1]

                        # 关键指标（字段名稳定）
                        result["roe"] = self._to_f(latest.get("净资产收益率"))
                        result["net_profit_margin"] = self._to_f(latest.get("销售净利率"))
                        result["gross_margin"] = self._to_f(latest.get("销售毛利率"))
                        result["current_ratio"] = self._to_f(latest.get("流动比率"))
                        result["debt_asset_ratio"] = self._to_f(latest.get("资产负债率"))
                        # 这些字段有时为 False/空，_to_f 会安全兜底为0.0
                        result["rev_growth"] = self._to_f(latest.get("营业总收入同比增长率"))
                        result["profit_growth"] = self._to_f(latest.get("净利润同比增长率"))

                        if 0 < result["debt_asset_ratio"] < 100:
                            result["leverage"] = round(1 / (1 - result["debt_asset_ratio"] / 100), 2)

                        result["_ok"]["finance"] = True
                        ths_success = True
                        break  # 成功获取，退出重试循环
                    else:
                        if attempt < self._spot_retry:
                            time.sleep(0.5 * (attempt + 1))
                            continue
                        result["_err"].append("ths_finance_empty")
                except Exception as e:
                    if attempt < self._spot_retry:
                        time.sleep(0.5 * (attempt + 1))
                        continue
                    result["_err"].append(f"ths_finance_error (尝试{attempt+1}/{self._spot_retry+1}): {type(e).__name__}: {e}")
            
            # 降级策略：如果THS接口失败，尝试其他财务接口
            if not ths_success:
                # 尝试使用其他财务接口作为降级
                try:
                    # 可以尝试其他接口，但这里先保持原有逻辑
                    pass
                except Exception as e:
                    result["_err"].append(f"finance_fallback_error: {type(e).__name__}: {e}")

            # 3. 若仍拿不到 ROE，则用 PB/PE 推算（标注为推算，不再默默写0）
            if not result["_ok"]["finance"] and result["pe_ttm"] > 0:
                result["roe"] = round((result["pb"] / result["pe_ttm"]) * 100, 2)
                # 只作为兜底推算，不写入 _ok.finance
                result["_err"].append("roe_estimated_by_pb_pe")

            # 4. 最终兜底：若估值字段已取得，补标 spot OK
            if not result["_ok"]["spot"]:
                if (result.get("pe_ttm", 0) or result.get("pb", 0) or result.get("total_mv", 0)):
                    result["_ok"]["spot"] = True

        except Exception as e:
            result["_err"].append(f"spot_error: {type(e).__name__}: {e}")
            print(f"⚠️ 财报异常: {e}")

        return result

    def _get_spot_df_cached(self, result: dict):
        """
        获取全市场 spot 数据（带缓存 + 重试）。
        工业级优化：增强重试机制和错误处理
        """
        now = time.time()
        if self._spot_cache_df is not None and (now - self._spot_cache_ts) < self._spot_cache_ttl_sec:
            return self._spot_cache_df

        last_err = None
        max_retries = max(1, self._spot_retry + 1)
        for i in range(max_retries):
            try:
                df = ak.stock_zh_a_spot_em()
                if df is None or df.empty:
                    raise RuntimeError("spot_df_empty")
                # 标准化代码列为6位
                code_col = next((c for c in df.columns if '代码' in c), None)
                if code_col:
                    df[code_col] = df[code_col].astype(str).str.zfill(6)
                self._spot_cache_df = df
                self._spot_cache_ts = now
                return df
            except Exception as e:
                last_err = e
                # 指数退避，降低瞬时波动/限流影响
                if i < max_retries - 1:
                    time.sleep(0.5 * (i + 1))  # 0.5s, 1s, 1.5s...

        if last_err is not None:
            result["_err"].append(f"spot_retry_failed (尝试{max_retries}次): {type(last_err).__name__}: {last_err}")
        return None

    def _get_indicator_snapshot(self, symbol: str, max_retries: int = 2):
        """
        估值指标兜底：尝试从 A 股指标接口提取 PE/PB/市值
        """
        symbol = str(symbol).strip().zfill(6)

        def _pick_col(df, keywords):
            cols = list(df.columns)
            for c in cols:
                c_low = str(c).lower()
                for kw in keywords:
                    kw_low = kw.lower()
                    if kw_low in c_low or kw in str(c):
                        return c
            return None

        fetchers = []
        if hasattr(ak, "stock_a_indicator_lg"):
            fetchers.append(ak.stock_a_indicator_lg)
        if hasattr(ak, "stock_zh_a_indicator"):
            fetchers.append(ak.stock_zh_a_indicator)
        if hasattr(ak, "stock_a_indicator"):
            fetchers.append(ak.stock_a_indicator)

        for fetch in fetchers:
            for attempt in range(max_retries):
                try:
                    df = fetch(symbol=symbol)
                    if df is None or df.empty:
                        continue
                    # 取最新记录
                    if "trade_date" in df.columns:
                        df = df.sort_values("trade_date")
                    latest = df.iloc[-1]
                    pe_col = _pick_col(df, ["pe_ttm", "市盈率", "PE", "TTM"])
                    pb_col = _pick_col(df, ["pb", "市净率", "PB"])
                    mv_col = _pick_col(df, ["total_mv", "总市值", "市值", "流通市值"])
                    pe_ttm = self._to_f(latest.get(pe_col)) if pe_col else None
                    pb = self._to_f(latest.get(pb_col)) if pb_col else None
                    total_mv = self._to_f(latest.get(mv_col)) if mv_col else None
                    ok = any([pe_ttm, pb, total_mv])
                    return {"pe_ttm": pe_ttm, "pb": pb, "total_mv": total_mv, "ok": ok}
                except Exception:
                    if attempt < max_retries - 1:
                        time.sleep(0.5 * (attempt + 1))
                        continue
        return None

    def _get_industry_board_names(self, max_retries: int = 2):
        now = time.time()
        if self._industry_board_names and (now - self._industry_board_ts) < self._industry_board_ttl_sec:
            return self._industry_board_names
        sources = []
        if hasattr(ak, "stock_board_industry_name_em"):
            sources.append(ak.stock_board_industry_name_em)
        if hasattr(ak, "stock_board_industry_name_ths"):
            sources.append(ak.stock_board_industry_name_ths)
        if not sources:
            return []
        for fetch in sources:
            for attempt in range(max_retries):
                try:
                    df = fetch()
                    if df is None or df.empty:
                        continue
                    name_col = next((c for c in df.columns if "板块" in c or "行业" in c or "名称" in c), None)
                    if not name_col:
                        name_col = df.columns[0]
                    names = [str(x).strip() for x in df[name_col].dropna().tolist() if str(x).strip()]
                    if names:
                        self._industry_board_names = names
                        self._industry_board_ts = now
                        return names
                except Exception:
                    if attempt < max_retries - 1:
                        time.sleep(0.3 * (attempt + 1))
                        continue
        return []

    def _normalize_industry_name(self, name: str):
        if not name:
            return ""
        n = str(name).strip()
        # 去掉括号内容（如 “计算机设备(申万)”）
        for sep in ["(", "（"]:
            if sep in n:
                n = n.split(sep)[0].strip()
        return n

    def _match_industry_board_name(self, industry: str):
        if not industry:
            return None
        industry = self._normalize_industry_name(industry)
        names = self._get_industry_board_names()
        if not names:
            return None
        # 优先完全匹配
        for name in names:
            if industry == self._normalize_industry_name(name):
                return name
        # 其次包含匹配
        for name in names:
            norm = self._normalize_industry_name(name)
            if industry in norm or norm in industry:
                return name
        return None

    def _lookup_industry_from_boards(self, symbol: str, max_retries: int = 2, max_scan: int = 200):
        symbol = str(symbol).strip().zfill(6)
        now = time.time()
        if symbol in self._industry_map and (now - self._industry_map_ts) < self._industry_map_ttl_sec:
            return self._industry_map[symbol]
        if not hasattr(ak, "stock_board_industry_cons_em"):
            return None

        names = self._get_industry_board_names()
        if not names:
            return None

        scanned = 0
        for name in names:
            scanned += 1
            if max_scan and scanned > max_scan:
                break
            codes = self._industry_cons_cache.get(name)
            if codes is None:
                cons_df = None
                for attempt in range(max_retries):
                    try:
                        cons_df = ak.stock_board_industry_cons_em(symbol=name)
                        if cons_df is not None and not cons_df.empty:
                            break
                    except Exception:
                        if attempt < max_retries - 1:
                            time.sleep(0.3 * (attempt + 1))
                            continue
                if cons_df is None or cons_df.empty:
                    continue
                code_col = next((c for c in cons_df.columns if "代码" in c or "证券代码" in c or "stock_code" in c), None)
                if not code_col:
                    continue
                codes = set(cons_df[code_col].astype(str).str.zfill(6).tolist())
                self._industry_cons_cache[name] = codes
            if symbol in codes:
                self._industry_map[symbol] = name
                self._industry_map_ts = now
                return name
        return None

    # ... (get_industry_peers, _find_val, _to_f 保持不变，直接复用原有的即可) ...
    # 为了完整性，这里简写保留辅助函数结构
    def get_industry_peers(self, symbol, max_retries=3):
        """
        工业级优化：获取行业和同行对比数据
        添加重试机制和降级策略
        """
        symbol = str(symbol).strip().zfill(6)
        if symbol in self._peers_cache:
            return self._peers_cache[symbol]

        industry = self._industry_cache.get(symbol)
        # 1) 个股信息接口（可能不稳定）- 添加重试机制
        if not industry:
            for attempt in range(max_retries):
                try:
                    info_df = ak.stock_individual_info_em(symbol=symbol)
                    if info_df is not None and not info_df.empty and "item" in info_df.columns:
                        row = info_df[info_df["item"].astype(str).str.contains("行业|所属行业")]
                        if not row.empty:
                            industry = str(row["value"].values[0]).strip()
                            break
                    if attempt < max_retries - 1:
                        time.sleep(0.5 * (attempt + 1))  # 指数退避
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(0.5 * (attempt + 1))
                        continue
                    print(f"⚠️ 获取个股信息失败 ({symbol}): {e}")
                    industry = None

        # 2) 使用缓存的全市场spot兜底
        dummy = {"_err": []}
        spot_df = self._get_spot_df_cached(dummy)
        if not industry and spot_df is not None and not spot_df.empty:
            code_col = next((c for c in spot_df.columns if '代码' in c), None)
            ind_col = next((c for c in spot_df.columns if '行业' in c), None)
            if code_col and ind_col:
                row = spot_df[spot_df[code_col].astype(str).str.zfill(6) == symbol]
                if not row.empty:
                    industry = str(row[ind_col].values[0]).strip()

        # 2.5) 行业兜底：从行业板块成分反查
        if not industry or industry == "未知":
            industry = self._lookup_industry_from_boards(symbol, max_retries=max_retries)

        # 3) 最后兜底：板块按代码前缀
        if not industry:
            prefix = symbol[:2]
            industry = {"60": "上海主板", "00": "深圳主板", "30": "创业板", "68": "科创板"}.get(prefix, "未知")

        # 4) 构建同行对比
        # 工业级优化：如果spot_df为空，尝试重新获取（带重试）
        try:
            if spot_df is None or spot_df.empty:
                # 尝试重新获取全市场数据（带重试）
                dummy = {"_err": []}
                full_market = self._get_spot_df_cached(dummy)
                if full_market is None or full_market.empty:
                    # 最后尝试直接调用（不带缓存）
                    for attempt in range(max_retries):
                        try:
                            full_market = ak.stock_zh_a_spot_em()
                            if full_market is not None and not full_market.empty:
                                break
                            if attempt < max_retries - 1:
                                time.sleep(0.5 * (attempt + 1))
                        except Exception as e:
                            if attempt < max_retries - 1:
                                time.sleep(0.5 * (attempt + 1))
                                continue
                            print(f"⚠️ 获取全市场数据失败 (尝试{attempt+1}/{max_retries}): {e}")
                    if full_market is None or full_market.empty:
                        return industry or "未知", pd.DataFrame()
            else:
                full_market = spot_df.copy()

            code_col = next((c for c in full_market.columns if '代码' in c), None)
            name_col = next((c for c in full_market.columns if '名称' in c), None)
            ind_col = next((c for c in full_market.columns if '行业' in c), None)
            mkt_cap_col = next((c for c in full_market.columns if '总市值' in c), None) or next((c for c in full_market.columns if '市值' in c), None)
            pe_col = next((c for c in full_market.columns if '市盈率' in c and '动' in c), None) or next((c for c in full_market.columns if '市盈率' in c), None)
            pb_col = next((c for c in full_market.columns if '市净率' in c), None)

            if code_col:
                full_market[code_col] = full_market[code_col].astype(str).str.zfill(6)

            peers_df = pd.DataFrame()
            
            # 优先尝试：通过行业名称获取该行业成分股 (修复：紫金矿业匹配银行问题)
            # 工业级优化：添加重试机制
            if industry and industry not in ["未知", "上海主板", "深圳主板", "创业板", "科创板"]:
                for attempt in range(max_retries):
                    try:
                        # 获取行业成分股代码列表
                        board_name = self._match_industry_board_name(industry) or industry
                        cons_df = None
                        if hasattr(ak, "stock_board_industry_cons_em"):
                            cons_df = ak.stock_board_industry_cons_em(symbol=board_name)
                        if (cons_df is None or cons_df.empty) and hasattr(ak, "stock_board_industry_cons_ths"):
                            cons_df = ak.stock_board_industry_cons_ths(symbol=board_name)
                        if (cons_df is None or cons_df.empty) and hasattr(ak, "stock_board_industry_cons_sina"):
                            cons_df = ak.stock_board_industry_cons_sina(symbol=board_name)
                        if cons_df is not None and not cons_df.empty:
                            cons_code_col = next((c for c in cons_df.columns if '代码' in c or '证券代码' in c or 'stock_code' in c), None)
                            if cons_code_col:
                                cons_codes = cons_df[cons_code_col].astype(str).str.zfill(6).tolist()
                                if cons_codes and code_col:
                                    peers_df = full_market[full_market[code_col].isin(cons_codes)].copy()
                                    if not peers_df.empty:
                                        break  # 成功获取，退出重试循环
                        if attempt < max_retries - 1:
                            time.sleep(0.5 * (attempt + 1))  # 指数退避
                    except Exception as e:
                        if attempt < max_retries - 1:
                            time.sleep(0.5 * (attempt + 1))
                            continue
                        print(f"⚠️ 获取行业成分股失败 ({industry}, 尝试{attempt+1}/{max_retries}): {e}")

            # 兜底1：如果 spot_df 自带行业列，且上面获取成分股失败
            if peers_df.empty and ind_col and industry not in ["未知", "上海主板", "深圳主板", "创业板", "科创板"]:
                norm_ind = self._normalize_industry_name(industry)
                peers_df = full_market[full_market[ind_col].astype(str).apply(self._normalize_industry_name) == norm_ind].copy()
            
            # 移除粗暴的板块前缀兜底，避免将紫金矿业（有色）匹配为市值最高的银行股
            # if peers_df.empty:
            #    peers_df = full_market[full_market[code_col].astype(str).str.startswith(symbol[:2])].copy()

            if peers_df.empty:
                # 如果找不到同行，尝试用全部A股的同名行业（如果spot里有行业列但没匹配上）
                if ind_col and industry:
                     peers_df = full_market[full_market[ind_col].astype(str).str.contains(self._normalize_industry_name(industry), na=False)].copy()
                
                if peers_df.empty:
                    return industry, pd.DataFrame()

            if mkt_cap_col:
                peers_df = peers_df.sort_values(by=mkt_cap_col, ascending=False).head(6).copy()
            else:
                peers_df = peers_df.head(6).copy()

            comparison_df = pd.DataFrame({
                "代码": peers_df[code_col].astype(str).str.zfill(6) if code_col else peers_df.index.astype(str),
                "名称": peers_df[name_col] if name_col else "",
                "PE(动)": peers_df[pe_col].apply(self._to_f) if pe_col else 0.0,
                "PB": peers_df[pb_col].apply(self._to_f) if pb_col else 0.0,
                "市值(亿)": (peers_df[mkt_cap_col].apply(self._to_f) / 100000000).round(2) if mkt_cap_col else 0.0
            })
            comparison_df['ROE(推算%)'] = np.where(comparison_df['PE(动)'] > 0,
                                                   (comparison_df['PB'] / comparison_df['PE(动)'] * 100).round(2), 0)

            self._industry_cache[symbol] = industry
            self._peers_cache[symbol] = (industry, comparison_df)
            return industry, comparison_df
        except Exception:
            return industry or "未知", pd.DataFrame()

    def _find_val(self, row, cols, keywords):
        for c in cols:
            if all(k in str(c) for k in keywords): return self._to_f(row[c])
        return 0.0

    def _to_f(self, val):
        try:
            if val is None or str(val) in ['-', 'nan', '']: return 0.0
            return float(str(val).replace('%', '').replace(',', ''))
        except:
            return 0.0