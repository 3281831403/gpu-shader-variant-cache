# gpu-shader-variant-cache

## ① 题面

加入着色器变体缓存后，同一材质反复编译、帧率周期性掉：变体键按插入顺序拼接，同一组特性换序就生成新键；键用内置 hash，跨进程结果不一致；缓存满时按 FIFO 淘汰，热点变体刚编译完就被踢；编译预算只数条目，忽略单个变体耗时，短帧仍然超时；失效只删除精确文件名，依赖头文件对应的变体漏删；特性合并冲突时静默取 a，覆盖结果不可预期；命中不更新访问时间，LRU 退化成 FIFO；容量不校验，超容后缓存无限增长；批量请求共用一个缓存句柄，请求之间状态串扰；capacity=0 不校验，计算槽位时除零。variant/cache.py 是核心，tests/test_cache.py 15个红。要修的是：键按排序后的特性内容生成稳定 SHA-256 键、严格 LRU 淘汰并保持容量上限、编译预算按毫秒累计且容差为 0、按依赖文件集合失效、冲突取 b 并显式标记、命中刷新最近使用时间、校验容量与键重复和空程序、批量请求隔离且任一非法整体回滚、capacity=0 抛 VariantError。

## ② 运行方式

在仓库根目录执行：

```bash
python -m pytest tests -q
```

初始状态输出尾部必须恰好为 `15 failed`：

```text
15 failed in <时间>s
```

15 个测试全部调用 `NotImplementedError` 桩，不出现 collection、import 或 syntax error。环境只使用标准库（`collections`、`hashlib`）与 pytest，离线、无随机数、无 sleep。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## ④ tests 不可修改

`tests/test_cache.py` 不可修改：不得删改测试、增加 skip/xfail、放宽断言，也不得在测试侧绕过失败。实现应落在 `variant/cache.py`，保留 `VariantError` 和 docstring 中键规范化、容量、编译预算与容差 0 的契约。
