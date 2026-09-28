# WebShop 持续学习消融汇总

*截至 2026-09-23。方法论与 `alfworld_summary.md` 一致（见该文件「消融配置方式」一节），
差异点与跨域对比见本文结论。*

## 环境与池子

- **WebShop**：12,087 个购物目标，agent 每轮从搜索结果/商品页选一个动作（`search[...]`、
  `click[...]`），买下商品即结束。官方 split 互不重叠：**test 500 / dev 1000 / train 10587**。
- **基础轨迹池**：`webshop_experiment/base_train_v1/`，train split 随机抽 200 题，无记忆。
- **分布外评测集**：`webshop_experiment/baseline_test57/`，test split 随机抽 57 题，与训练池零重叠。
- 评测参数全网格一致：`max_steps=15`、`seed=20260822`、`--memory-top-k 3`、`max-parallel 4`。
- **环境是独立常驻服务**（`scripts/webshop_env_server.py`，端口 3100），这是三个域里唯一不在
  进程内跑环境的。rollout 侧只是它的 HTTP 客户端，跑在仓库 `.venv`；只有 env server 需要
  `webshop_venv`（装了 spacy，且 spacy 3.3.0 要求 pydantic<1.9，与 openai 不兼容——
  两者必须留在不同 venv 里）。

## WebShop 独有的两个测量特点

1. **除二元 `success` 外还有连续 `score`（0-1）**，按商品类型/属性/选项/价格的匹配度给分。
   本文两列都报，因为**它们会给出不同的排序**（见结论 3）。
2. **episode 极短**：中位 8 步、4,032 字符，对比 ALFWorld 29 步 / 10,814 字符、
   ScienceWorld 30 步 / 27,724 字符。200 题墙钟 14 分钟 vs 59 / 20 分钟。
   这不是跑漏了——任务形态就是「搜索 → 点商品 → 选属性 → 下单」。

## 基线：失败模式与另外两个域都不同

```
train200  pass=0.3800 (76/200)  score=0.6347   purchased 184 | max_steps 16
test57    pass=0.3684 (21/57)   score=0.6712   purchased  54 | max_steps  3
```

**92% 的任务正常完成购买**，agent 几乎总能买到东西，只是买得不完全对。且成功与失败的局长度
**完全一样**（都是中位 4 个 agent 动作）——不是「没耐心」，是**选错了**。

三域失败模式对照：ALFWorld 是「没做完」（超步数），ScienceWorld 是「一步走死」
（`focus on` 指错 −100 分），WebShop 是「做完了但不够准」。

## 三个建池臂

| 臂 | 路由结果 | 产物 |
|---|---|---|
| router（`llm`） | `{neither 145, sft 47, memory 7}` | 活跃记忆 **5 条**；47 候选 → **11 条验证通过**（yield 0.234） |
| force_memory | `{memory 198}` | 活跃记忆 **182 条** |
| force_sft | `{sft 198}`，活跃记忆 **0** | 198 候选 → **98 条验证通过**（yield 0.495） |

## ⚠️ 主结果已改用完整 test split（500 题）

本文最初的结论基于 test split 的 57 题随机子样本（为与另外两个 benchmark 的规模对齐）。
之后补跑了**官方完整 test split 的全部 500 题**，六个配置全跑，**推翻了其中两条基于 57 题的判断**。

环境经实测是**逐题确定性的**：57 题在独立运行与 500 题运行中 success/reward/步数 **57/57 完全一致**，
所以差异不是噪声，是 57 题的抽样偏差——±2 题即 ±3.5pp，撑不起配置之间几个百分点的比较。

### 完整 test 500（本文的主结果）

| 配置 | memory | SFT | pass_rate | mean_score | timeout | vs 基线（二元 / 连续分） |
|---|---|---|---|---|---|---|
| 无记忆基线 | — | 底座 | 0.3940 (197/500) | 0.6574 | 40 | — |
| router 仅记忆 | 5 条 | 底座 | 0.3840 (192/500) | 0.6501 | 38 | 33 升 38 降 (−5) / 62 升 68 降 |
| **force_memory** | **182 条** | 底座 | **0.3280 (164/500)** | **0.5251** | **136** | 19 升 52 降 (**−33**) / 31 升 **125** 降 |
| router 仅 SFT | — | 11 条 | 0.4020 (201/500) | 0.6921 | **14** | 28 升 24 降 (+4) / 69 升 44 降 |
| router 记忆+SFT | 5 条 | 11 条 | 0.4220 (211/500) | 0.6972 | 18 | 36 升 22 降 (+14) / 78 升 36 降 |
| **force_sft** | — | **98 条** | **0.4420 (221/500)** | 0.6868 | 37 | 43 升 19 降 (**+24**) / 74 升 39 降 |

### 被 500 题推翻的两条 57 题结论

| | 57 题的结论 | 500 题的事实 |
|---|---|---|
| SFT 样本量 | router 11 条 (0.4035) **优于** force_sft 98 条 (0.3684)，「样本多但不更好」 | **反过来**：98 条 +24 题，11 条只 +4 题，**样本越多越好**（与 ALFWorld 同向，与 ScienceWorld 不同） |
| 记忆叠加 | 叠加 −1 题，「记忆只带来伤害」 | **反过来**：叠加相对仅 SFT **+10 题**，5 条精选记忆是有益补充 |

**教训（对后续 task stream）**：为跨 benchmark 规模统一而抽的小样本，只够支撑「有无大效应」
（如 force_memory 的 −33），不够支撑「哪个配置更好」这类几个百分点的比较。凡是要比较配置高低，
用完整 split。

### 原 57 题子样本（保留，仅用于三域规模统一的横向对比）

| 配置 | memory | SFT | pass_rate | mean_score | timeout | vs 基线（二元 / 连续分） |
|---|---|---|---|---|---|---|
| 无记忆基线 | — | 底座 | 0.3684 (21/57) | 0.6712 | 3 | — |
| router 仅记忆 | 5 条 | 底座 | 0.3333 (19/57) | 0.6601 | 3 | 0 升 2 降 |
| **force_memory** | **182 条** | 底座 | **0.2982 (17/57)** | **0.5105** | **17** | 1 升 5 降 / 2 升 14 降 |
| **router 仅 SFT** | — | **11 条** | **0.4035 (23/57)** | **0.7151** | **0** | 3 升 1 降 / 7 升 5 降 |
| router 记忆+SFT | 5 条 | 11 条 | 0.3860 (22/57) | 0.7129 | 2 | — |
| force_sft | — | 98 条 | 0.3684 (21/57) | **0.7057** | 1 | 4 升 4 降 / **10 升 5 降** |

## 训练集内（train 200 题，按配置量化泄露）

| 配置 | 整体 | 泄露 | 去泄露（本配置/同子集基线） | mean_score | timeout |
|---|---|---|---|---|---|
| 无记忆基线 | 0.3800 (76/200) | 0 | — | 0.6347 | 16 |
| router 记忆 5 条 | 0.3850 (77/200) | 0 | 0.3850 / 0.3800 | 0.6388 | 19 |
| force_memory 182 条 | 0.2850 (57/200) | 0 | **0.2850 / 0.3800** | 0.4705 | **67** |
| router 仅 SFT 11 条 | 0.4050 (81/200) | 11 | 0.3704 / 0.3968 | 0.6811 | 8 |
| router 记忆+SFT | 0.4200 (84/200) | 11 | 0.3862 / 0.3968 | 0.6885 | 7 |
| force_sft 98 条 | 0.5050 (101/200) | **98** | 0.0490 / 0.0098 ※ | 0.6975 | 18 |

※ 98/200 泄露后只剩 102 题，同子集基线只有 1 题成功（0.0098），样本有强选择偏差，
**这个格子的分布内数字不可用**。`baseline -> force_sft` 的 25 升 **0 降**同样主要是背诵。

---

## 四条结论

### 1. WebShop 是三个域里唯一「记忆净有害」的，且与步数预算无关

```
                      train200          完整 test500
force_memory 182 条   −19 题 (0.285)    −33 题 (0.328)，连续分 31 升 125 降
router 记忆   5 条     +1 题 (噪声)      −5 题 (33 升 38 降，噪声)
```

超时数随注入量单调上升：test500 上基线 40 → router 5 条 38 → force_memory 182 条 **136**
（27% 的任务跑不完）。这是整个三域实验里最大的单一负效应。

**做了步数预算对照排除替代解释**（`baseline_test57_steps25` / `forcemem_test57_steps25`）：

```
                     pass      score    timeout
基线 @15 步         0.3684    0.6712      3
基线 @25 步         0.3684    0.6931      1      ← 基线本就不受步数约束
force_mem @15 步    0.2982    0.5105     17
force_mem @25 步    0.3158    0.5398     15      ← 放宽 67% 步数只回收 1 题
```

把步数从 15 放宽到 25，force_memory 的超时只从 17 降到 15、pass 只回收 1 题。如果损害真来自
「记忆挤占步数」，25 步该让大部分超时任务完成。**所以不是预算被挤占，是记忆内容本身把 agent
带进了低效的搜索路径**——连续分 2 升 14 降更直接：即使买成了也买得更不对。

**为什么这个域相反**：另外两个域的失败是「不知道怎么做」，记忆能补；WebShop 的失败是「选错了」，
而未筛选的记忆提供的正是**别的商品的选择经验**，直接构成误导。

### 2. SFT 完全不占 prompt 预算，这在短 episode 域里是决定性的

test500 的超时数：基线 40、router 仅 SFT **14**、force_memory **136**。SFT 把知识写进权重，
一个 token 的 prompt 都不占；记忆走检索，注入多少就挤占多少。`alfworld_summary.md` 结论 1 说
「检索位是稀缺资源」，WebShop 表明在中位 8 步的域里稀缺的是**整个 episode 预算**，而 SFT 绕开了
这个约束。

三个域的 SFT 侧全部正向（ALFWorld +13 题、ScienceWorld +5 题、WebShop **+24 题**），
记忆侧则是两正一负。

### 3. 二元 pass_rate 和连续 score 给出不同排序，只看前者会漏掉真实效果

在 57 题子样本上，`force_sft` 的二元 pass 与基线完全打平（21/57）而连续分明显更高
（0.7057 vs 0.6712，10 升 5 降）——只看 `pass_rate` 会误判成「SFT 无效」。

500 题上两个指标不再矛盾（pass +24 题、连续分 74 升 39 降），但连续分仍然更敏感：
`router 仅 SFT` 的 pass 只 +4 题（28 升 24 降，接近抵消），连续分却是 **69 升 44 降**，
明确正向。**凡是有连续评分的 task stream，都应两列都报**——二元指标会把「更接近正确」
这类改善整个丢掉。

### 4. 「SFT 样本越多越好」在这里**成立**（与 ALFWorld 同向，与 ScienceWorld 相反）

```
完整 test500：router 11 条 (0.4020, +4 题)  <  force_sft 98 条 (0.4420, +24 题)
```

**这条曾被 57 题子样本读反**（那上面是 0.4035 > 0.3684，据此写过「样本多但不更好」）。
完整 split 上样本量从 11 提到 98 带来 6 倍的增益。

与 `scienceworld_summary.md` 结论 2 对照：那里 48 条不如 13 条，是因为 ScienceWorld 的样本
几乎全是**修复**类（168 个失败任务里只救回 19 个，修复率 11.3%）；WebShop 的 98 条里有 75 条
**巩固**类（巩固率 100%）。所以决定项仍是样本构成，只是 WebShop 的构成偏向了能泛化的那一类。

### 附：教师修复率——又一个反例

| | 修复率（原本失败→救回） | 巩固率（原本成功→整理） |
|---|---|---|
| AppWorld | 62.5% | — |
| ALFWorld | 60.0% | 94.7% |
| **WebShop** | **18.7%** (23/123) | **100.0%** (75/75) |
| ScienceWorld | 11.3% | 90.6% |

`scienceworld_summary.md` 结论 3 已推翻「教师修复机制跨域普适」，WebShop 再添一例：
**四个域里只有两个达到 60%，另外两个都在 20% 以下**。而巩固率四个域全部 90% 以上——
教师整理成功流程的能力是普适的，救回失败的能力不是。

WebShop 的 100% 巩固率也解释得通：成功轨迹就是 4 步的干净流程，教师几乎不可能整理错。

---

## 产物与参数

### 建池
- router：`webshop_experiment/router_llm_probe_v1/`（bank 5 条、sft_pool 11 条）
- force_memory：`webshop_experiment/router_force_memory_v1/`（bank 182 条）
- force_sft：`webshop_experiment/router_force_sft_v1/`（bank 0 条、sft_pool 98 条）

### SFT LoRA（`scripts/train_agent_sft_lora_peft.py`）

| | 样本 | loss（3 epoch 均值） | merged |
|---|---|---|---|
| router | 11 | 0.1247 → 0.0242 → 0.0079 | `/nas04/yixuh/ws_router_merged` |
| force_sft | 98 | 0.0651 → 0.0205 → 0.0057 | `/nas04/yixuh/ws_force_sft_merged` |

merge 后逐张量校验：目标模块 rel 3.01e-03 / 6.19e-03，vision 塔与融合专家权重逐位未变。

### 评测
```
freeze_replay_routermem_200 / routermem_test57       router 5 条记忆 + 底座
freeze_replay_forcemem_200  / forcemem_test57        force_memory 182 条 + 底座
routersftonly_train200      / routersftonly_test57   空 bank + router SFT 模型
routerboth_train200         / routerboth_test57      5 条记忆 + router SFT 模型
freeze_replay_forcesft_200  / forcesft_test57        空 bank + force_sft 模型
baseline_test57_steps25 / forcemem_test57_steps25    步数预算对照（max_steps=25）
```

### 过程中修的一件事
给 `webshop_venv` 装 openai 时升级了 pydantic 到 2.x，直接弄坏了 spacy（env server 依赖，
重启即失败）。已回滚到 pydantic 1.8.2。正确的分工代码里本就写明：
`router_sft_pipeline._REPLAY_CONFIG["webshop"]` 注释说 rollout 侧跑在仓库 `.venv`、
只有 env server 需要 `webshop_venv`。
