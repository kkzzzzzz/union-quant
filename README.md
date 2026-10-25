# Union Quant — 统一量化核心项目

将原有的三个子工程（`index_fetcher`、`strategy`、`now_asset`）整合收敛为**一个项目、一个库表、一个拉数据、一个算策略**的轻量、极易扩展的后端核心链路。

---

## 一、整体架构

```text
union/
├── sql/
│   └── schema.sql                  # 【一个库表】统一幂等 Schema (行情估值+因子窄表+信号单表+多租户台账)
├── configs/
│   ├── assets.yaml                 # 统一标的与数据源字典 (指数/债券/黄金/ETF/股债性价比SBE)
│   └── strategies/                 # 统一声明式策略配置 (YAML)
│       ├── global_asset_allocation.yaml
│       ├── nasdaq_dynamic_position.yaml
│       ├── etf_dual_momentum.yaml
│       ├── etf_core_rotation.yaml
│       ├── size_style_rotation.yaml
│       └── sbe_notice.yaml
├── core/
│   ├── config.py                   # 全局配置加载器
│   ├── db.py                       # 统一 Supabase 网关 + 本地离线缓存
│   ├── notifier.py                 # 统一多通道消息推送 (钉钉/飞书/企微/TG/Server酱/PushPlus)
│   ├── data_hub/                   # 【一个拉数据】多源行情与估值同步中心
│   │   ├── models.py               # MarketBar / SyncReport 标准模型
│   │   ├── providers.py            # 新浪实时 / 好买估值 / 中债官网 / AkShare(ETF+黄金) / SBE采集
│   │   └── sync.py                 # 日常增量同步 + 晚间估值追赶补齐 + 历史回溯 + T+1矩阵加载
│   └── quant_engine/               # 【一个算策略】向量化因子、触发器、策略与回测引擎
│       ├── factors.py              # @register_factor 纯函数因子注册表与批量/按需计算服务
│       ├── triggers.py             # 可组合调仓触发器 (CalendarTrigger / DriftTrigger / ExtremumTrigger)
│       ├── strategies.py           # @register_strategy 统一策略实现 (动态配置/双动量/阈值轮动/SBE/等权)
│       ├── backtester.py           # 严格 T+1 向量化/事件混合回测与绩效评估引擎
│       └── pipeline.py             # 统一回测与每日信号入库推送流水线
├── tests/
│   └── test_unified_system.py      # 全链路离线自动化单元测试
└── cli.py                          # 统一命令行入口
```

---

## 二、快速开始

### 1. 查看已注册资产、因子与策略
```bash
python cli.py list
```

### 2. 拉取数据与计算因子（一个拉数据）
```bash
# 首次全量拉取历史行情与10年历史估值
python cli.py sync --mode history --start 2015-01-01 --compute-factors

# 每日定时增量同步最新行情与估值（自动检测估值滞后并分拆日期存储）
python cli.py sync --mode daily --compute-factors --notify

# 晚间专项重拉估值（补齐好买晚于收盘更新的 PE/PB/股息率）
python cli.py sync --mode catchup --compute-factors
```

### 3. 运行策略回测与每日信号（一个算策略）
```bash
# 运行单个或全部策略的 T+1 历史回测
python cli.py backtest --strategy global_asset_allocation
python cli.py backtest --all

# 生成每日实盘信号（严格物理截断 T 日收盘数据）并写入 strategy_signals 表 + 推送通知
python cli.py signal --all --mode live

# 本地演练模式（不写远程表、不发外部通知）
python cli.py signal --strategy nasdaq_dynamic_position --mode dry-run --no-notify
```

---

## 三、如何扩展新因子与新策略

1. **新增一个因子**：在 [`core/quant_engine/factors.py`](file:///d:/personal/union/core/quant_engine/factors.py) 中添加带 `@register_factor("your_factor_id")` 装饰器的向量化函数（仅需 5~10 行代码），无需再维护数十个 JSON 配置文件。
2. **新增一个标的**：在 [`configs/assets.yaml`](file:///d:/personal/union/configs/assets.yaml) 中添加一行标的代号及数据源映射即可。
3. **新增一个策略**：在 [`configs/strategies/`](file:///d:/personal/union/configs/strategies) 下新建一个 YAML 文件，或通过 `@register_strategy("your_type")` 在 [`core/quant_engine/strategies.py`](file:///d:/personal/union/core/quant_engine/strategies.py) 注册新算法类。
