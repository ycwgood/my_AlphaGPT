多只股票的特征数据计算不会冲突吗？

好问题！现在我看到实际代码后，发现了潜在的数据冲突问题。

当前的数据处理方式
```
# 合并所有股票数据
df_merged = pd.concat(all_dfs, ignore_index=True)

# 特征工程
train_df, features = FeatureEngineer.create_features(train_df_raw)
test_df, _ = FeatureEngineer.create_features(test_df_raw)
```
存在的问题
1. 时间序列连续性被破坏 ⚠️
当合并多只股票后，时间序列不再连续：
```
股票A: 2024-01-01, 2024-01-02, 2024-01-03, ...
股票B: 2024-01-01, 2024-01-02, 2024-01-03, ...
合并后: A_01-01, A_01-02, B_01-01, B_01-02, ...
```
2. 移动窗口特征计算错误
如果特征包含移动平均、动量等时间序列计算：
```
# 假设特征计算需要前5天数据
momentum_5 = (close[-5] - close[0]) / close[0]

# 在合并数据中，close[-5] 可能来自另一只股票！❌
```
3. 跨股票污染
```
# 示例：如果按顺序存储
df_merged = [A股票全部行, B股票全部行]
# 计算 momentum 时，B股票的前几行会引用A股票的数据
```
建议修改方案
```
# 在 main() 函数中，特征工程部分修改如下

# 按股票分别进行特征工程
train_dfs = []
test_dfs = []

for code in config.codes:
    # 按股票代码分别处理
    train_df_code = train_df_raw[train_df_raw['code'] == code].copy()
    test_df_code = test_df_raw[test_df_raw['code'] == code].copy()
    
    # 按股票分别进行特征计算（保持时间序列连续）
    train_df_code, features = FeatureEngineer.create_features(train_df_code)
    test_df_code, _ = FeatureEngineer.create_features(test_df_code)
    
    train_dfs.append(train_df_code)
    test_dfs.append(test_df_code)

# 特征工程后再合并
train_df = pd.concat(train_dfs, ignore_index=True)
test_df = pd.concat(test_dfs, ignore_index=True)

features = [f for f in features if f in train_df.columns and f in test_df.columns]
train_df = train_df[['trade_date', 'code', 'close'] + features].copy()
test_df = test_df[['trade_date', 'code', 'close'] + features].copy()
```
核心区别
方面	当前方式	建议方式
特征计算顺序	合并后统一计算	每只股票独立计算
时间序列连续性	❌ 被破坏	✅ 保持连续
移动窗口准确性	❌ 跨股票污染	✅ 准确无误
计算复杂度	略低	略高
建议立即修改，否则多股票的回测结果可能不准确。

