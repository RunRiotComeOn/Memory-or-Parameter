# ScienceWorld 持续学习消融汇总

*截至 2026-09-23。方法论与 `alfworld_summary.md` 完全一致，逐条对照见该文件「消融配置方式」一节。
本文件只记 ScienceWorld 的数字、与 ALFWorld 的差异，以及这些差异改写了哪些此前的结论。*

## 环境与池子

- **ScienceWorld**：30 个任务类型（煮沸、导电性、孟德尔遗传、斜面摩擦……），train 3592 /
  dev 1796 / test 1819 个变体。task_id 形如 `<task_name>::<variation_id>`，三个 split 是同一批
  任务名下互不相交的变体号区间——所以 `test` 就是这里的分布外线，对应 ALFWorld 的 `valid_unseen`。
- **基础轨迹池**：`scienceworld_experiment/base_train_v1/`，train split 分层抽样 200 题，覆盖全部
  30 个任务类型，无记忆。
- **分布外评测集**：`scienceworld_experiment/baseline_test57/`，test split 分层抽样 57 题，
  覆盖 30 个任务类型，与训练池**零重叠**。
- 评测参数全网格一致：`max_steps=30`、`seed=20260822`、`--memory-top-k 3`、`max-parallel 4`、
  temperature 0、thinking disabled。task agent 为 Qwen3.5-35B-A3B，确定性 serving。

## 开跑前修掉的 harness 问题（都是实测出来的）

1. **动作列表无上限**。`_ordered_actions` 只对 connect/disconnect 做字符预算，其余动作全量放行；
   注释里「~93 项 / ~1,660 字符」的测量在物件密集房间不成立——一个 `grow-plant` 回合产生了
   **66,547 字符**的单轮文本，12 题探针里 **3 题（25%）死于 `context_overflow`**。这是工具限制
   被计成了 agent 失败。
2. **第一版修法反而掉了成功**。改成「总预算 + 按动词轮转采样」后 overflow 消失，但 `find-plant`
   从 `focus on adult pea plant`（解出）退化成 `focus on pea plant`（名字不精确、被环境拒绝）——
   因为我同时在 prompt 里鼓励「列表没有就自己构造命令」，而 `focus on` 恰是一击致命的动词。
   **该 prompt 改动已回退**。
3. **最终方案：只有最近 2 轮携带完整动作列表，更早的历史剥掉**（`_strip_history`）。当前轮看得全
   （12,000 字符预算，实测各房间够用），历史不再把上下文撑爆。剥离**同时作用于 `steps` 记录**，
   因为 SFT 样本是从 trajectory 建的，记录必须与模型实际看到的文本一致。
4. **nudge 改成可执行的**：原来只说「那不是合法动作」，agent 会抱着同一个 near-miss 连撞三次被
   判死（`pick up glass cup`、`go to outside`）。现在直接列出最接近的合法动作。

验收（100 题分层探针）：`context_overflow` **25% → 0**，`ungrounded_action` 回到 4%。

**步数上限维持 30，有据**：超时任务的中位得分只有 25、仅 4/37 达到 75 分以上——是迷路而非差一点，
翻倍步数预计只救回 4-6 题却让网格每次运行贵一倍。真正的瓶颈是下面那条。

## 基线：任务难得多，且失败模式高度集中

```
train200  pass_rate = 0.1600 (32/200)   task_ended 95 | max_steps 61 | ungrounded 12 | solved 32
test57    pass_rate = 0.1579 (9/57)     task_ended 31 | max_steps 12 | ungrounded  5 | solved  9
```

对比 ALFWorld 的 0.69 / 0.7544。**47.5% 的任务死于 `task_ended`，中位得分 −100**——即
`focus on` 指错对象导致的一击致命。ScienceWorld 的 `focus on` 是评分承诺而非查看动作。

## 三个建池臂

| 臂 | 路由结果 | 产物 |
|---|---|---|
| router（`llm`） | `{neither 155, sft 42, memory 1, both 2}` | **活跃记忆 2 条**；44 候选 replay → **13 条验证通过**（yield 0.295） |
| force_memory | `{memory 200}` | **活跃记忆 159 条**（存活率 79.5%） |
| force_sft | `{sft 200}`，活跃记忆 **0** | 200 候选 replay → **48 条验证通过**（yield 0.24） |

**router 的记忆臂几乎是空的（2 条）**，不是 ALFWorld 那种「写了很多被 refine 合并掉」，而是压根没写：
168 个失败轨迹里多数是 `focus on` 猜错后 3 步就结束的残局，router 判断无可提炼，选了 `neither`。

## 主结果：分布外（test 57 题，无泄露）

| 配置 | memory | SFT 模型 | pass_rate | vs 基线（逐题） |
|---|---|---|---|---|
| 无记忆基线 | — | 底座 | 0.1579 (9/57) | — |
| router 仅记忆 | 2 条 | 底座 | 0.1754 (10/57) | +1 题（噪声） |
| **force_memory** | **159 条** | 底座 | **0.2456 (14/57)** | 7 升 2 降 (+5) |
| **router 仅 SFT** | — | **13 条** | **0.2456 (14/57)** | 9 升 4 降 (+5) |
| router 记忆+SFT | 2 条 | 13 条 | 0.2456 (14/57) | 与仅 SFT 持平 |
| force_sft | — | 48 条 | 0.2281 (13/57) | — |

```
router sft(13)  -> force sft(48)  :  5 升  6 降   net  -1
router sft(13)  -> router mem+sft :  1 升  1 降   net  +0
force mem(159)  -> router sft(13) :  8 升  8 降   net  +0
```

## 训练集内（train 200 题，按配置量化泄露）

| 配置 | 整体 | 训练时见过自己轨迹 | 去泄露（本配置/同子集基线） |
|---|---|---|---|
| 无记忆基线 | 0.1600 (32/200) | 0 | — |
| router 记忆 2 条 | 0.1550 (31/200) | 0 | 0.1550 / 0.1600 |
| force_memory 159 条 | 0.2050 (41/200) | 0 | 0.2050 / 0.1600 |
| router 仅 SFT 13 条 | 0.2400 (48/200) | 13 | 0.2086 / 0.1444 |
| router 记忆+SFT | 0.2500 (50/200) | 13 | 0.2193 / 0.1444 |
| force_sft 48 条 | 0.2650 (53/200) | **48** | 0.1316 / 0.0197 ※ |

```
baseline        -> force mem(159) : 12 升  3 降   net  +9
baseline        -> router sft(13) : 26 升 10 降   net +16
router sft(13)  -> router mem+sft :  2 升  0 降   net  +2
router sft(13)  -> force sft(48)  : 15 升 10 降   net  +5
```

※ 去掉 48 题泄露后剩 152 题，但那是「连教师计划都救不回来」的最难子集，同子集基线只有 3 题成功
（0.0197），增幅看着巨大却有强选择偏差。`force_sft` 的可信证据是分布外的 0.2281。

---

## 四条结论，其中两条改写了 ALFWorld 的判断

### 1. router 筛选的价值依赖基础轨迹池的质量（新增，`alfworld_summary.md` 结论 1 的边界条件）

```
ALFWorld      基线 0.69：router 34 条 (0.8947)  >  强制全写 143 条 (0.8246)   筛选赢
ScienceWorld  基线 0.16：router  2 条 (0.1754)  <<  强制全写 159 条 (0.2456)   筛选输到只剩噪声
```

ALFWorld 的结论是「更少但更准，优于更多但不筛」。ScienceWorld 表明这句话有前提：**素材本身要够好**。
基线 0.16 时，失败轨迹多数是 3 步内自杀的残局，router 如实判断「无可提炼」，于是 200 题只写 3 次。
筛选没有筛出精华，是筛到了什么都不剩。

**对后续 task stream 的操作含义**：先看基础池的 pass_rate。偏低时，记忆侧应当直接用强制全写建立
厚度，router 筛选留到基础能力足以产出可提炼素材之后再启用。

### 2. SFT 侧「越多越好」不成立——取决于样本是修复还是巩固（改写 ALFWorld 结论 1 的后半句）

```
ALFWorld      21 条 (0.8596)  <  181 条 (0.9825)   7 升 0 降，数量完胜
ScienceWorld  13 条 (0.2456)  >   48 条 (0.2281)   5 升 6 降，数量反而略输
```

ALFWorld 的解释是「SFT 不占检索位，多训没有副作用」。机制没错，但样本构成才是决定项：

| | 巩固样本 | 修复样本 |
|---|---|---|
| ALFWorld force_sft(181) | 138 | 61 |
| ScienceWorld force_sft(48) | 29 | **19** |

ScienceWorld 的 200 个候选里 168 个来自失败任务，救回来的只有 19 个。修复样本教的是「这个特定错误
的特定修法」，巩固样本教的是「这类任务的通用流程」——后者才是能泛化的那种。基线低 → 巩固素材少 →
即使样本数更多，泛化收益也不增反降。

### 3. 教师修复机制**不是**跨域普适的（推翻 running_log §15 的推测）

真正可比的「修复率」（原本失败的任务，教师计划让学生重跑成功的比例）：

```
AppWorld      62.5%  (running_log §11)
ALFWorld      60.0%  (running_log §15)
ScienceWorld  11.3%  (19/168)        ← force_sft 臂
巩固率对照     90.6%  (29/32)         ← 同一批次，说明不是管线坏了
```

running_log §15 把 AppWorld 62.5% 与 ALFWorld 60.0% 的接近称为「教师模型修复机制有跨域普适性最像
样的证据」。ScienceWorld 的 11.3% 推翻了这个推广。同一次运行里巩固率高达 90.6%，排除了管线故障：
教师能把成功轨迹整理成可复现的流程，但在 ScienceWorld 上几乎无法把失败救回来。合理的解释是这里的
主导失败（`focus on` 指错对象，−100 分立即结束）一旦发生就无法在同一回合内补救，计划能讲清「该先
探索再 focus」，但学生仍要自己在几十个同名物件里挑对那一个。

### 4. 记忆和 SFT 修的不是同一批题

分布外相对基线各自修好的题目集合：

```
force_mem(159) 修好 7 题 ∩ router_sft(13) 修好 9 题 → 仅 2 题重叠
force_mem(159) ∩ force_sft(48)  → 7/6 题中 3 题重叠
router_sft(13) ∩ force_sft(48)  → 9/6 题中 3 题重叠
```

三个配置都落在 14/57 不是因为都只够到最容易的题，而是各自啃下了不同的失败模式。这也解释了为什么
叠加在分布内是正向的（`router_sft → router_mem+sft` 为 2 升 0 降），尽管这里 bank 只有 2 条、
叠加效应基本测不出来。

---

## 各配置的产物与参数

### 建池

- router：`scienceworld_experiment/router_llm_probe_v1/`（bank 2 条、sft_pool 13 条）
- force_memory：`scienceworld_experiment/router_force_memory_v1/`（bank 159 条）
- force_sft：`scienceworld_experiment/router_force_sft_v1/`（bank 0 条、sft_pool 48 条）

### SFT LoRA（`scripts/train_agent_sft_lora_peft.py`，transformers+peft 双卡 + 梯度检查点）

| | 样本 | loss（3 个 epoch 均值） | merged |
|---|---|---|---|
| router | 13 | 0.3217 → 0.1442 → 0.0873 | `/nas04/yixuh/sw_router_merged` |
| force_sft | 48 | 0.2314 → 0.0934 → 0.0452 | `/nas04/yixuh/sw_force_sft_merged` |

起始 loss 远高于 ALFWorld（那边 0.045）——ScienceWorld 的成功轨迹对模型是真正的新东西。
merge 后逐张量校验：目标模块 rel 3.13e-03 / 5.05e-03，vision 塔与融合专家权重逐位未变。

### 途中修掉的一个训练侧真 bug（也影响 ALFWorld 的结果）

`build_example` 的超长截断原本保留尾部、丢弃前缀——**会把系统提示整个切掉**，训练样本不再以
system 轮开头，与推理时的 prompt 形状完全不符。ScienceWorld 样本 5,777–8,070 tokens，原先 6,144
的上限让一半以上样本中招；ALFWorld 那批最长 8,462，也有少数受影响（p90 仅 3,885，面很小）。
已改成**保留系统轮 + 尾部**，默认上限提到 12,288（实测显存天花板是 16,384，24,576 OOM）。
**ALFWorld 的网格未因此重跑**，其 `force_sft(181)` 结果带有这一已知瑕疵。

### 评测

```
freeze_replay_routermem_200 / routermem_test57          router 2 条记忆 + 底座
freeze_replay_forcemem_200  / forcemem_test57           force_memory 159 条 + 底座
routersftonly_train200      / routersftonly_test57      空 bank + router SFT 模型
routerboth_train200         / routerboth_test57         2 条记忆 + router SFT 模型
freeze_replay_forcesft_200  / forcesft_test57           空 bank + force_sft 模型
```
