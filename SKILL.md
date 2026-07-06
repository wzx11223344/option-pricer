---
name: option-pricer
description: 五模型期权定价引擎（BS/二叉树/蒙特卡洛/Heston/Merton）+ 复步长自动希腊字母 + 波动率曲面 + Plotly 看板。支持 CLI 命令式调用与 Python API。适用于期权定价、希腊字母计算、隐含波动率求解、波动率曲面构建。触发词：期权定价、BS公式、希腊字母、隐含波动率、波动率曲面、Heston、Merton跳跃。
---

# 期权定价引擎

## 能力边界

### ✅ 支持的能力
- **五模型定价**：Black-Scholes 解析解、Binomial Tree (CRR)、Monte Carlo 仿真、Heston 随机波动率、Merton 跳跃扩散
- **自动希腊字母**：基于复步长微分（complex-step differentiation），对任意定价函数自动计算 Delta / Gamma / Vega / Theta / Rho，无需手动推导公式
- **隐含波动率求解**：Newton-Raphson 方法从市场报价反推 IV，支持看涨/看跌期权
- **波动率曲面**：从真实市场数据（如 akshare 拉取 50ETF 期权链）构建波动率微笑/曲面
- **模型对比**：一键对比多个模型对同一期权的定价结果
- **Plotly 交互看板**：可视化损益图、希腊字母曲面、波动率曲面
- **CLI 与 Python API 双模**：命令行一键定价 + Python 库级调用

### ❌ 不支持的能力
- 美式/百慕大/亚式等奇异期权定价
- 实时交易接口与订单执行
- 高频数据处理或 Level 2 行情解析
- 回测引擎或策略框架
- 外汇/固收/信用衍生品定价

## 触发条件

当用户提及以下关键词或意图时，应优先调用本 Skill：
- "期权定价"、"计算期权价格"、"BS公式"、"Black-Scholes"
- "计算希腊字母"、"Delta"、"Gamma"、"Vega"、"Theta"、"Rho"
- "隐含波动率"、"IV"、"implied volatility"、"反推波动率"
- "波动率微笑"、"波动率曲面"、"volatility surface"
- "Heston模型"、"随机波动率"、"Merton跳跃扩散"、"蒙特卡洛定价"
- "对比不同模型的期权价格"、"模型对比"
- "二叉树定价"、"CRR模型"

## 使用方法

### CLI 快速使用
```bash
# 单模型定价
python pricing_cli.py price --model bs --S 100 --K 105 --T 0.5 --r 0.03 --sigma 0.2

# 计算希腊字母
python pricing_cli.py greeks --model bs --S 100 --K 105 --T 0.5 --r 0.03 --sigma 0.2

# 隐含波动率
python pricing_cli.py iv --price 3.5 --S 100 --K 105 --T 0.5 --r 0.03

# 波动率曲面（从真实数据）
python pricing_cli.py surface --ticker 510050

# 模型对比
python pricing_cli.py compare --models bs,heston,merton --S 100 --K 105 --T 0.3

# Plotly 交互看板
python pricing_cli.py dashboard --S 100 --K 105 --T 0.5 --r 0.03 --sigma 0.2
```

### Python API 调用
```python
from option_pricer.models import BlackScholes, BinomialTree, MonteCarlo, Heston, Merton
from option_pricer.greeks import compute_cs_greeks, all_greeks

# Black-Scholes 定价
price = BlackScholes.call_price(S=100, K=105, T=0.5, r=0.03, sigma=0.2)

# 复步长自动希腊字母
greeks = all_greeks(BlackScholes.call_price, S=100, K=105, T=0.5, r=0.03, sigma=0.2)

# Heston 模型
heston_price = Heston.call_price(S=100, K=105, T=0.5, r=0.03, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, v0=0.04)
```

## 输出示例

CLI 定价输出：
```
Model: Black-Scholes
Call Price: 4.0829
Put Price:  6.6152
Implied Vol: 0.2000
```

希腊字母输出：
```
Delta: 0.5632 | Gamma: 0.0421 | Vega: 19.8742
Theta: -4.9213 | Rho: 25.1134
```

## FAQ

**Q: 复步长微分和有限差分有什么不同？**
A: 复步长微分使用 f'(x) = Im[f(x+ih)] / h，避免有限差分的相减抵消问题，精度可达 O(h^2) 且步长可极小而不会出现数值不稳定。

**Q: 支持哪些期权类型？**
A: 当前所有模型均支持欧式看涨/看跌期权。Binomial Tree 模型可通过增大步数逼近美式期权价格，但不支持显式美式提前行权判定。

**Q: 数据从哪里获取？**
A: 波动率曲面功能通过 akshare 拉取真实市场期权链数据（如 510050 50ETF 期权）。定价模型本身无需外部数据。

**Q: Heston 模型的 5 个参数如何校准？**
A: 可通过市场期权报价使用最小二乘法（least_squares）校准 Heston 参数。模型模块提供了参数校准接口。

**Q: 蒙特卡洛仿真的精度如何控制？**
A: 通过 paths 参数控制路径数量，默认 100,000 条路径。可用 antithetic variates 对偶变量法降低方差，收敛速度约为 O(1/sqrt(N))。

## 技术栈

- **核心依赖**: NumPy, SciPy
- **可视化**: Plotly
- **数据获取**: akshare（国内期权链数据）
- **实现方式**: 纯 NumPy/SciPy 数值计算，无 QuantLib 依赖
- **语言**: Python 3.8+
