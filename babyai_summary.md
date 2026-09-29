# BabyAI 持续学习消融汇总

*截至 2026-09-28。方法论与 `alfworld_summary.md` 完全一致，逐条对照见该文件「消融配置方式」一节。
本文件只记 BabyAI 的数字、与前五个域的差异，以及这些差异改写了哪些此前的结论。
BabyAI 是第一个接入的 **AgentGym** 环境，客户端按 AgentGym 的统一 HTTP 契约写
（`/create` `/reset` `/step` `/observation` `/close`），不是按 BabyAI 写的——所以接下一个
AgentGym 环境只需要新的 writer prompt，harness 基本可复用。*

## 环境与池子

- **BabyAI**：文本化的网格世界导航 + 物体操作。40 个 level（`GoToRedBall`、`Open`、
  `PickupLoc`……）共用一套动作词表。`data_idx` 决定任务：服务端取
  `all_levels[data_idx % 40 + 1]`、`seed = data_idx // 40`，所以**世界完全由 data_idx 决定**，
  没有服务端级别的随机种子。
- **切分按 layout seed**（`data_idx // 40`），不是按 level：两侧都出现全部 40 个 level，但布局
  互不相交。因此这是**同分布留出集**，和 ScienceWorld / WebShop 的 test 同类，**不是**
  ALFWorld `valid_unseen` 那种跨环境泛化测试。
- **基础轨迹池**：`babyai_experiment/base_train_v1/`，seeds 0–19 抽样 200 题，覆盖全部 40 个
  level，无记忆。pass 0.3700（74/200）、score 0.3425。
- **留出评测集**：`babyai_experiment/baseline_test80/`，seeds 20–21 全部 80 题（40 个 level × 2），
  与训练池零重叠。
- 评测参数全网格一致：`max_steps=15`（计 **agent 回合**，不是网格内原始步数——高层动作
  `go to red ball 1` 一回合可能消耗多个原始步，服务端自己另有 50 步上限）、`seed=20260822`、
  `--memory-top-k 3`、`max-parallel 4`、temperature 0、thinking disabled。
  task agent 为 Qwen3.5-35B-A3B，确定性 serving。

### 两个 BabyAI 特有的测量决定

1. **`success` 的判据是「结束且拿到任何分」，不是满分。** 奖励按消耗的原始步数折扣，实测已解出
   的 episode 落在 0.83–0.99，**从不等于 1.0**，而抓错物体恰好是 0。用 `score >= 1.0` 会让每一次
   成功都被记成失败，整张网格失去分辨率。
2. **`score` 和 `pass_rate` 都报。** BabyAI 的官方指标是连续奖励，pass_rate 是本项目为了跨域可比
   另加的。两列在本文件所有表格里同时出现，结论只在两列同向时才下。

## 开跑前修掉的 harness 问题

1. **上游 `environment.py:319` 用 `np.cross` 算二维向量叉积**，numpy 2.0 已移除该用法，
   环境服务端起不来。`babyai_venv` 内固定 numpy<2。
2. **测试集原本只有 57 题，且没有任何记录说明为什么**。`split_task_ids("test")` 返回 800–879 共
   80 个 id，磁盘上的 `baseline_test57` 只覆盖其中 57 个，缺的 23 个集中在高 level 段。逐个试过，
   **它们在服务端都能正常 reset**，不是坏样本。本轮把留出集改回完整的 80 题并重测基线：
   pass 从 0.3333 降到 0.2875——那 23 题确实更难，原来的 57 题子样本是偏乐观的。
   `baseline_test57` 保留在磁盘上，但本文件所有数字都基于 80 题。
3. **probe 里的「世界一致性」检查调用了不存在的接口**（`BabyAIEnvClient.info()["seed"]`）。
   AgentGym 没有 `/info` 路由，BabyAI 也没有服务端种子。改成**观测值比对**：reset 一个已记录的
   任务，要求首个观测与轨迹里记录的逐字节相同——比原本想比的种子号是更强的保证。
4. **驱动脚本里残留着 tau2 的参数**。`build_arm` 传了 `--domain`（BabyAI 的 probe 没有这个 flag）、
   `--train-rollout` 指向不存在的路径；`eval_one` 同样传了 `--domain` 且漏了必填的
   `--experiment-name`。前者让 phase A 的三条臂全部以 argparse 报错秒退，后者让五个 eval 全部
   秒退。**而脚本照常跑完并打印了一张空表**——下游每一步的「跳过」条件都挂在失败臂没写出的产物上，
   于是什么都不报错，只是什么都没做。现在 phase A 失败即 `exit 1`。
   这是本项目第三次遇到同一类 bug，见文末。

## 基线：失败模式高度单一

| | solved | pass_rate | mean_score | 终止原因 |
|---|---|---|---|---|
| train 池（200 题） | 74/200 | 0.3700 | 0.3425 | max_steps 占 63% |
| 留出 test（80 题） | 23/80 | 0.2875 | 0.2673 | max_steps=57、solved=23 |

0.2875 是六个域里**最低的基线**。失败几乎全部是 `max_steps`——没有一次 `ungrounded_action`
（动作都是从列表里抄的、格式合法），也没有 `context_overflow`。

## 三个建池臂（均在 200 题 train 池上）

| 臂 | 路由分布 | 记忆库 | SFT 池 |
|---|---|---|---|
| router（`--router-mode llm`） | neither 165 / sft 28 / memory **3** | 3 条 | 28 replay → **7** 条通过（yield 0.25） |
| force_memory | memory 195 / null 5 | 154 条 | 0（定义如此） |
| force_sft | sft 200 | 0（定义如此） | 200 replay → **92** 条通过（yield 0.46） |

- `null` 是草稿写失败，全部发生在 base agent 成功的题上。
- **force_sft 的 rescue / consolidation 拆分**：原本失败的 126 题救回 **26（20.6%）**，
  原本成功的 74 题巩固住 **66（89.2%）**。
- **router 的筛选在这里几乎没有增益**：它挑出的 28 题里 25 题原本是失败的，验证通过率 0.25，
  而 force_sft 对全部 126 道失败题的 rescue 率是 0.206。**挑过的题并不比没挑的更容易被救回来。**

## 主结果：留出集（test 80 题，无泄露）

| 配置 | solved | pass_rate | mean_score | 相对 baseline |
|---|---|---|---|---|
| baseline | 23/80 | 0.2875 | 0.2673 | — |
| router memory（3 条） | 25/80 | 0.3125 | 0.2929 | +2 |
| force_memory（154 条） | 32/80 | 0.4000 | 0.3724 | +9 |
| **router SFT only（7 条）** | **43/80** | **0.5375** | **0.5038** | **+20** |
| router memory + SFT | 42/80 | 0.5250 | 0.4907 | +19 |
| force_sft（92 条） | 39/80 | 0.4875 | 0.4557 | +16 |

全部 80 题完成、0 errors。`pass_rate` 与 `mean_score` 两列排序完全一致。

**逐题增减**（比净值信息量大得多）：

| 配置 | 新解出 | 解不出了 | 净 |
|---|---|---|---|
| force_memory | 14 | **5** | +9 |
| router SFT | 20 | **0** | +20 |
| force_sft | 17 | 1 | +16 |

- **router SFT 是唯一严格优于 baseline 的配置**：baseline 解出的 23 题一题没丢。
- **force_memory 弄坏了 5 题**——检索到的记忆会把原本能做对的题带偏。这和 WebShop 的
  「记忆净有害」是同一类现象，只是在 BabyAI 上没有压过它的增益。

## 训练集内（train 200 题，按配置量化泄露）

五个冻结配置在**建池用的那 200 道题**上重跑（`scripts/run_babyai_indist_replay.sh`）。
全部 200 题完成、0 errors。

> 实际执行细节：phase D 结束后只删了磁盘上的 merged 模型目录，vLLM 服务并没有停，权重仍在显存里。
> 所以 phase M 重建出来的模型没有被加载——`serve()` 发现 `/v1/models` 返回的路径与请求一致就复用了
> 原有服务。重建因此是多余的，但有两个副产品：它证明了 merge 可复现（校验值与原轮逐位相同，
> 见「产物与参数」），并且**留出集与训练集内的数字来自同一个模型实例**，可比性最强。

| 配置 | train200 pass | score | test80 pass | score |
|---|---|---|---|---|
| baseline（池子来源） | 0.3700 | 0.3425 | 0.2875 | 0.2673 |
| router memory | 0.3400 | 0.3189 | 0.3125 | 0.2929 |
| force_memory | 0.4250 | 0.3968 | 0.4000 | 0.3724 |
| router SFT | 0.5150 | 0.4822 | **0.5375** | **0.5038** |
| router both | 0.4900 | 0.4616 | 0.5250 | 0.4907 |
| force_sft | **0.5550** | **0.5222** | 0.4875 | 0.4557 |

注意 **train 上的排名与 test 相反**：force_sft 在 train 上最高（0.5550 > 0.5150），
到 test 上反转（0.4875 < 0.5375）。

### 聚合数字会掩盖泄露——必须按「是否进过训练池」拆开

train200 不是同质的一批：force_sft 的 92 条样本全部来自这 200 道题，所以这个数字是
「92 道见过的」和「108 道没见过的」的混合。在同一次运行内拆开：

| | 进过池子 | 同批题的 baseline | 没进池子 | 同批题的 baseline |
|---|---|---|---|---|
| **force_sft**（92 条池） | **90/92 = 0.9783** | 66/92 = 0.7174 | 21/108 = 0.1944 | 8/108 = 0.0741 |
| **router SFT**（7 条池） | 5/7 = 0.7143 | 1/7 = 0.1429 | 98/193 = 0.5078 | 73/193 = 0.3782 |

- **force_sft 在自己训练过的题上几乎满分（97.8%）**——LoRA 把池子训到 loss 0.0000 的后果在这里
  直接可见。它在 train 上对 router SFT 的全部领先都来自这 92 道背下来的题。
- **router SFT 的池子只有 7 道**，同样背得很牢（0.7143 vs baseline 0.1429），但 7 道对 200 的
  聚合几乎没有影响，所以它的 train 数字基本是干净的。
- 池子本身是偏易的：force_sft 的 92 道里 baseline 已经做对 66 道——这个 66 与建池时的
  consolidation 计数完全吻合（66 巩固 / 26 修复）。router 的 7 道里 baseline 只做对 1 道，
  与它主要挑失败题一致。

**方法论教训**：只报 train 池的聚合 pass_rate 会把「背下 92 道题」读成「泛化变强」。
前几个域的同名小节都只报了聚合值，本域是第一次做池内 / 池外拆分——`alfworld_summary.md`、
`scienceworld_summary.md`、`webshop_summary.md` 的对应小节应当补做同样的拆分后再对比。
（本轮未回头重跑那三个域。）

### 池外的增益不能与 test80 直接比

force_sft 池外增益 +0.1204、router SFT 池外增益 +0.1295，都低于各自在 test80 上的增益
（+0.20 / +0.25）。这**不是**「对没见过的训练题更差」——池外那批是残渣：base agent 失败
**且** guided replay 也没救回来的题，baseline 只有 0.0741 / 0.3782，难度与 test80 的混合分布
不可比。要判断泛化，仍以 test80 为准。

## 结论

### 1. 主要失败模式是「原地打转」，SFT 买到的是打破重复，不是更好的路线

先排除掉一个错误假设：我原本以为 SFT 的增益来自「改用高层动作 `go to X` 而不是逐格挪，
从而在 15 回合预算内完成」。**数据不支持**——SFT 之后高层动作占比反而从 54.4% 降到 41.3%，
`turns/solved` 也从 2.70 升到 4.02。

真正的机制在重复率上：

| 配置 | 打转的 episode（某动作连续重复 ≥5 次） | 动作多样性（distinct/总数） |
|---|---|---|
| baseline | **51.2%** | 0.577 |
| router memory（3 条） | 52.5% | 0.580 |
| force_memory（154 条） | 38.8% | 0.597 |
| router SFT | **27.5%** | 0.687 |
| router both | 31.2% | 0.660 |
| force_sft | 33.8% | 0.694 |

baseline 有一半以上的 episode 卡在重复同一个动作直到耗尽预算。SFT 把它砍掉近一半，记忆砍掉约
四分之一，而只有 3 条记忆的 router memory 完全没动（52.5%，与 baseline 无异）。
`turns/solved` 上升说明 SFT 模型解出的是**原本超时的那批更难的题**，不是把原来会做的题做得更快。

**这解释了为什么 7 条样本够用**：被修正的是一个行为层面的退化习惯（卡住就重复），
不是任务特定的知识。所以样本量与增益不成比例——这是「SFT 修什么」这个问题在本项目里
第一次有了明确的机制答案。

### 2. 「样本越多越好」在这里再次不成立，而且方向比 ScienceWorld 更极端

> **边界条件（2026-09-29 补，由 TextCraft 推翻后收紧）**：本结论**仅在任务不可组合时成立**。
> TextCraft 的两个 SFT 池同样以巩固样本为主（199 条池中 94% 是巩固），结果却是**越多越好**
> ——199 条在组合泛化线上 +19，18 条只有 +4，与这里的方向完全相反。所以决定因素不是
> 「巩固 vs 修复」的比例，而是**样本携带的知识能否跨实例复用**：TextCraft 的巩固样本教的是
> 「配方树如何自底向上分解」这一程序，换个目标仍然适用；BabyAI 的巩固样本教的是某张地图
> 怎么走，换张图就作废，多了只会互相稀释。详见 `textcraft_summary.md` 结论 2。

7 条样本的 router SFT（0.5375）**高于** 92 条样本的 force_sft（0.4875）。逐题看，
router SFT 独占 5 题、force_sft 独占 1 题、共享 38 题——不是噪声性的互有胜负。

这与 `scienceworld_summary.md` 结论 2（「SFT 侧越多越好不成立，取决于样本是修复还是巩固」）
同向但更强：force_sft 的 92 条里有 66 条是**巩固**（原本就成功），只有 26 条是**修复**；
router 的 7 条则来自它主动挑出的失败题。巩固样本在这里似乎不只是无用，而是会把打破重复
这个信号稀释掉。

**注意**：router 臂的筛选本身（见上文）并没有提高 rescue 率——0.25 vs 0.206 基本持平。
所以增益不是来自「挑得准」，而是来自**池子的构成比例**（7 条里 6 条是修复 vs 92 条里 72% 是巩固，
两者都由训练集内的池内 baseline 证实：1/7 vs 66/92）。这两件事需要区分开，否则会高估路由的价值。

训练集内的拆分还给出了第二个机制：force_sft 在 train 上高于 router SFT，**全部**来自它背下的
92 道题（池内 0.9783）。也就是说「92 条 vs 7 条」的差距在训练集上表现为记忆、在留出集上表现为
劣势——多出来的 85 条巩固样本买到的是记忆化，不是能力。

### 3. 记忆修的题几乎是 SFT 修的题的子集——与 ScienceWorld 结论 4 相反

`scienceworld_summary.md` 结论 4 是「记忆和 SFT 修的不是同一批题」。BabyAI 上不成立：

- force_memory 新解出的 14 题里，**12 题**（86%）也被 router SFT 解出；
- 两者增益集合的并集是 22 题，而 SFT 单独就有 20 题——**记忆只额外贡献 2 题**。

机制上讲得通：如果两条通道修的是同一个缺陷（打破重复），那它们自然会命中同一批题。
`router memory + SFT`（0.5250）低于 `router SFT` 单独（0.5375）也与此一致——叠加不仅不相加，
3 条记忆挤占的 prompt 预算还小幅拖累了结果。

### 4. 路由写记忆的意愿随基线质量单调上升——BabyAI 提供了这条曲线的下端点

router 在 200 题里只写了 **3 条**记忆（1.5%），是六个域里最低的；BabyAI 的基线 0.2875 也是
六个域里最低的。接上 `tau2_summary.md` 结论 1 的受控证据（tau2 三个子域 45% / 24% / 17%
对应基线 0.96 / 0.74 / 0.69），这条单调关系现在在 0.29 处有了端点。

实际后果：**router memory 这一格在 BabyAI 上等于没有干预**（+2 题，0 条记忆生效的
episode 占绝大多数，打转率与 baseline 持平）。要判断记忆在本域有没有用，只能看
force_memory 那一格。

### 5. 教师修复率的两极分化，第六个域

| 域 | rescue（原本失败） | consolidation（原本成功） |
|---|---|---|
| AppWorld | 62.5% | ≥85% |
| ALFWorld | 60.0% | ≥85% |
| tau2-airline | 66.7% | ≥85% |
| **BabyAI** | **20.6%** | **89.2%** |
| WebShop | 18.7% | ≥85% |
| ScienceWorld | 11.3% | ≥85% |

BabyAI 明确落在低档，与 WebShop 几乎重合。consolidation 率 89.2% 继续维持「哪里都 ≥85%」的
规律——六个域无一例外。

## 产物与参数

### 建池
- `scripts/run_babyai_router_llm_probe.py`，三条臂分别 `--router-mode llm|force_memory|force_sft`，
  `--sft-writer teacher|none|teacher`。teacher 为 Gemini，key 在
  `/nas04/yixuh/.config/continual-memory/gemini_api_key`。
- writer prompt：`src/trajectory_memory_lab/babyai_sft_writer.py`。域特有的三点约束：动作必须
  带颜色+类型+序号的精确字符串、高层动作严格优于逐格路线（奖励按原始步数折扣）、
  失败通常是「没找到」或「认错物体」而非「多步流程执行错」。
- 产物：`babyai_experiment/babyai_{router_probe,force_memory,force_sft}/`。

### SFT LoRA（`scripts/train_agent_sft_lora_peft.py`，transformers+peft 双卡 + 梯度检查点）
- router 池 7 条 → 21 步 ×3 epoch；force_sft 池 92 条 → 276 步 ×3 epoch，
  末段 loss 0.0000 / grad_norm 0.000（**池子被完全记住**）。
- 两个 adapter 经 `scripts/rekey_lora_to_full_model.py` 改键后 merge。merge 校验（相对变化）：

  | LoRA | `[target]` 语言模型层 | `[untouched]` vision tower |
  |---|---|---|
  | router（7 条） | 2.46e-03 | 0.00e+00 |
  | force_sft（92 条） | 5.79e-03 | 0.00e+00 |

  改动幅度与池子大小同向，vision tower 两次都逐位未变。

### 驱动脚本
- `scripts/run_babyai_full_grid.sh`：phase 0（80 题基线）→ A（三条建池臂，一臂一个 replica 并行）
  → B/C（两次 LoRA + 改键 + merge，GPU 6,7）→ D（五个 eval，三 replica 并行）→ 报表。
  全程一次跑完，rc=0，merged 模型用完即删。
- `scripts/run_babyai_indist_replay.sh`：从存活的 adapter 重建 merged 模型，在建池用的那 200 题
  上重跑五个配置，用于量化泄露。
- GPU 2、3 属于另一位用户，全程未触碰。

### 「不报错、只是少做事」——第三次

前两次记在 `tau2_summary.md`（bank 路径写错→九个 eval 各自 40 题全报错却照常写出
`pass_rate: null` 的 summary，被跳过检查当成已完成；LoRA 被静默 CPU offload）。
本轮第三次：**phase A 三条臂全部以 argparse 错误秒退，而驱动脚本照常走完 B/C/D 并打印了一张空表**
——因为下游每一步的跳过条件都挂在失败臂没写出的产物上。

共同特征：失败不抛异常，只是让后续步骤少做事，而「少做事」和「已完成」在产物层面长得一样。
现在三处都已加上显式断言（bank 必须存在、`pass_rate` 必须非 null、phase A 失败即 `exit 1`）。
