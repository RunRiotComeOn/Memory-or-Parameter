# TextCraft 持续学习消融汇总

*截至 2026-09-29。方法论与 `alfworld_summary.md` 完全一致，逐条对照见该文件「消融配置方式」一节。
本文件只记 TextCraft 的数字、与前六个域的差异，以及这些差异改写了哪些此前的结论。
TextCraft 是第二个接入的 **AgentGym** 环境，客户端已抽成共享的
`agentgym_client.AgentGymEnvClient`（BabyAI 改为只提供默认端口的子类，行为不变），
所以接下一个 AgentGym 环境只需要新的 writer prompt 和 agent 系统提示。*

## 环境与池子

- **TextCraft**：文本化的 Minecraft 合成游戏。每局给一张配方表和一个目标物品，agent 要把目标
  拆成子目标、取基础材料、自底向上合成。动作只有三种：`craft <产物> using <原料>`、
  `get <数量> <物品>`（**仅限没有配方的基础材料**）、`inventory`。
- **动作是生成的，不是从列表里选的**。没有 ALFWorld / BabyAI 那样的 `Available actions` 块，
  不合语法的回复一律得到 "Could not execute ..."，所以 `extract_command` 校验的是文法而非菜单。
- **每局混入最多 10 条与目标无关的干扰配方**，「哪些配方相关」本身是任务的一部分。
- **奖励是二值的**（目标物品入库即 1 并终止），所以 `mean_score` 与 `pass_rate` 恒等；
  两列仍然都报，以保持跨域表格统一——某次运行两列不相等就意味着环境变了。

### 难度是一个显式参数，这是本域独有的

`data_idx` 索引的是**按配方树深度排序**的目标列表，所以深度切分是真正的组合泛化测试：

| 深度 | 目标数 | baseline |
|---|---|---|
| 1 | 132 | 1.00 |
| 2 | 285 | 0.92 |
| 3 | 116 | 0.66 |
| 4 | 11 | 0.18 |

难度在 depth 2→3 之间有一道**悬崖**。三个切分（`textcraft_agent.split_task_ids`，
种子 20260928，全部记录在 `task_manifest.json`）：

- **train 200**（depth 1–2 分层抽样）：所有池子的来源。baseline **0.940**（188/200）。
- **test 80**（depth 1–2，与 train 不相交）：同分布留出。baseline **0.925**（74/80）。
  已接近天花板，**它的作用是损害探测器，不是看增益**。
- **deep 127**（depth 3–4 全部）：任何池子都没见过。**主结果线**，baseline **0.622**（79/127）。

### 池子构成：BabyAI 的镜像

train 池 baseline 0.94 意味着 **188 道巩固 / 12 道修复**，而 BabyAI 是 74 巩固 / 126 修复。
这不是缺陷而是设计：主线是 deep，池子只在浅层积累，正好用来检验
`babyai_summary.md` 结论 2（「巩固样本买到的是记忆化而不是能力」）能否跨域。
**这条预测在本域被推翻了，见结论 2。**

## 评测参数

全网格一致：`max_steps=40`、`seed=20260822`、`--memory-top-k 3`、`max-parallel 6`、
temperature 0、thinking disabled，task agent 为 Qwen3.5-35B-A3B 确定性 serving。

**`max_steps=40` 来自实测对照，不是拍脑袋**：在 20 轮上限下 6 道 depth-3 任务失败，
把同样 6 道放到 40 轮重跑，**2 道成功**（第 21、31 轮），另外 4 道仍失败。
也就是说 20 轮在截断真正会做的题，测的是预算而不是推理；40 轮下观察到的成功都在 31 轮内收敛。
代价是 depth 3 的 baseline 从 0.33 升到 0.66——这个幅度比 6 题对照预估的（约 0.50）更大，
小样本对照低估了效应量。

## 三个建池臂（均在 200 题 train 池上）

| 臂 | 路由分布 | 记忆库 | SFT 池 |
|---|---|---|---|
| router（`--router-mode llm`） | neither 156 / memory 23 / sft 15 / both 3 | **24** 条 | 18 replay → **18** 条通过（yield 1.00） |
| force_memory | memory 198 / null 2 | **175** 条（六域最大） | 0（定义如此） |
| force_sft | sft 200 | 0（定义如此） | 200 replay → **199** 条通过（yield 0.995） |

- **教师 rescue 率 12/12 = 100%**，consolidation 187/188 = 99.5%，都是六个域里最高的。
- **router 放过了三分之二的可修复失败**：12 道失败题里它只挑了 4 道做 sft，7 道判 `neither`、
  1 道判 `memory`；而 force_sft 对那 12 道的 rescue 率是 100%。
- 写记忆比例 **13%**（26/200 含 both），对比 BabyAI 的 1.5%。两域 train 基线分别是 0.94 和 0.2875，
  与 `tau2_summary.md` 结论 1 的单调关系一致。

## 主结果一：deep 127 题（组合泛化，depth 3–4）

| 配置 | solved | pass_rate | mean_score | depth 3 | depth 4 | vs baseline |
|---|---|---|---|---|---|---|
| baseline | 79/127 | 0.6220 | 0.6220 | 77/116 | 2/11 | — |
| router memory（24 条） | 78/127 | 0.6142 | 0.6142 | 77/116 | 1/11 | −1 |
| **force_memory（175 条）** | 68/127 | **0.5354** | 0.5354 | 67/116 | 1/11 | **−11** |
| router SFT（18 条） | 83/127 | 0.6535 | 0.6535 | 79/116 | 4/11 | +4 |
| router memory + SFT | 82/127 | 0.6457 | 0.6457 | 81/116 | 1/11 | +3 |
| **force_sft（199 条）** | **98/127** | **0.7717** | 0.7717 | 93/116 | 5/11 | **+19** |

全部 127 题完成、0 errors。

**逐题增减**（比净值信息量大得多）：

| 配置 | 新解出 | 解不出了 | 净 |
|---|---|---|---|
| router memory（24） | 11 | 12 | −1 |
| force_memory（175） | 5 | **16** | −11 |
| router SFT（18） | 18 | 14 | +4 |
| router both | 14 | 11 | +3 |
| **force_sft（199）** | **24** | **5** | **+19** |

除 force_sft 外，每个配置都在大量「换题」——解出一批同时弄坏一批。
**force_sft 是唯一不破坏已有能力的配置**（只丢 5 道）。

## 主结果二：test 80 题（同分布，depth 1–2）

| 配置 | solved | pass_rate | vs baseline |
|---|---|---|---|
| baseline | 74/80 | 0.9250 | — |
| router memory + SFT | 77/80 | 0.9625 | +3 |
| router SFT（18 条） | 78/80 | 0.9750 | +4 |
| force_memory（175 条） | 78/80 | 0.9750 | +4 |
| router memory（24 条） | 79/80 | 0.9875 | +5 |
| force_sft（199 条） | **80/80** | **1.0000** | +6 |

全部为正、**无一造成损害**，与 BabyAI 的 force_memory 掉 5 道相反。但六格全挤在 0.925–1.000 的
6 道题区间内，**这条线没有分辨率**，作用只是确认没有损害。注意 force_memory 在这条线上 +4、
在 deep 上 −11——同一个记忆库，方向完全相反。

## 结论

### 1. 记忆在组合泛化上是净有害的，且呈剂量效应

| 记忆库 | deep 净变化 | 解不出了 |
|---|---|---|
| 24 条（router） | −1 | 12 |
| 175 条（force） | **−11** | **16** |

条数越多，害处越大。机制在 episode 长度上可见：

| 配置 | turns/episode | max_steps 失败 |
|---|---|---|
| baseline | 26.5 | 48 |
| force_memory | **28.3** | **59** |
| force_sft | **20.4** | **29** |

记忆库里装的全是 depth 1–2 的经验，检索到深层目标上就是误导：agent 按浅层配方模式走弯路，
episode 变长、耗尽 40 轮预算。而 SFT 把 episode 缩短了 6 轮，说明它学到的是更直接的分解路径。

这是 `webshop_summary.md` 结论 1（「记忆净有害」）最锐利的一个版本，并且第一次有了**剂量-反应**
证据和**同一记忆库在两条线上方向相反**的对照（test +4 / deep −11）。
记忆的适用范围因此可以收紧为：**检索内容与目标同分布时有用，跨难度层级时有害**。

### 2. 「样本越多越好」在这里成立，与 BabyAI 完全相反——边界条件是任务可组合性

| | SFT 池 | 池中巩固占比 | 主线净变化 |
|---|---|---|---|
| BabyAI | 7 条 | 少 | **+20**（vs 92 条的 +16） |
| BabyAI | 92 条 | 72% | +16 |
| **TextCraft** | **18 条** | 多 | **+4** |
| **TextCraft** | **199 条** | **94%** | **+19** |

BabyAI 的结论 2 是「巩固样本买到的是记忆化而不是能力，多了会稀释信号」。本域**两个池子都是
巩固为主，结果却是越多越好**，所以那条结论的决定因素不是巩固/修复比例。

更可能的区分点是**任务是否可组合**：TextCraft 的浅层样本教的是「配方树如何自底向上分解」这一
程序，深层目标是同一程序的更多层嵌套，所以样本越多覆盖的配方模式越广、迁移越强；BabyAI 的
样本教的是具体某张地图怎么走，换张图就没用，多了只会互相稀释。

**`babyai_summary.md` 结论 2 需要加边界条件**：「仅在任务不可组合（样本知识不能跨实例复用）时
成立」。这与当初给 ALFWorld 结论 1 补边界条件是同一类修订。

### 3. 路由筛选在本域是明确的净损失

router 在 12 道可修复失败里只挑了 4 道，deep 线上 +4；force_sft 全收，+19。
逐题看 force_sft 独占 21 题、router SFT 独占 6 题、共享 77 题——不是噪声性互有胜负。

与 BabyAI 对照（那里 router 的 7 条打赢了 force 的 92 条）可知：**筛选的价值不是普适的，
它取决于被丢弃的样本是否可迁移**。在可组合的域里，丢掉的巩固样本仍然携带程序知识，筛选就是
纯损失；在不可组合的域里，它们是噪声，筛掉才有收益。

### 4. 教师 rescue 率的两极分化，可能取决于任务信息对教师是否完全可见

| 域 | rescue | consolidation |
|---|---|---|
| **TextCraft** | **12/12 = 100%** | 99.5% |
| tau2-airline | 66.7% | ≥85% |
| AppWorld | 62.5% | ≥85% |
| ALFWorld | 60.0% | ≥85% |
| BabyAI | 20.6% | 89.2% |
| WebShop | 18.7% | ≥85% |
| ScienceWorld | 11.3% | ≥85% |

TextCraft 的 100%（n=12，95% 区间约到 0.74）是最高的。机制上说得通：**整棵配方树就印在观测里**，
教师掌握的信息与 agent 完全相同且足以解题；而 WebShop（商品可得性）、ScienceWorld（长流程）
都有教师看不到的隐藏信息。这给那条分化规律提供了一个候选解释——**分化取决于任务信息是否对
教师完全可见，而非任务难度**——七个域的数据与之一致，但尚未做受控检验。

consolidation 率 99.5% 继续维持「哪里都 ≥85%」的规律，七个域无一例外。

## 产物与参数

### 环境
- `agentenv-textcraft`（AgentGym）在 :36002，tmux `textcraft_env`，`/nas04/yixuh/textcraft_venv`。
  服务必须从 `agentenv-textcraft/` 目录启动，否则 `minecraft_dir="agentenv_textcraft/"` 找不到配方。
- `textcraft_experiment/task_manifest.json`：544 个目标的 `data_idx →（目标, 深度）`映射，
  已与实时服务在每个深度边界逐一核对。AgentGym 没有 `/info`、TextCraft 没有服务端种子
  （目标是 `data_idx` 的纯函数），所以一致性检查用的是**观测逐字节比对**
  （`AgentGymEnvClient.assert_reproduces`）。

### 建池
- `scripts/run_textcraft_router_llm_probe.py`，三条臂分别 `--router-mode llm|force_memory|force_sft`。
  teacher 为 Gemini，key 在 `/nas04/yixuh/.config/continual-memory/gemini_api_key`。
- writer prompt：`src/trajectory_memory_lab/textcraft_sft_writer.py`。域特有的四点约束：
  可迁移的是子目标链而非动作序列、数量算术是最常见的可修复错误、`get` 只对无配方物品有效、
  配方表里混有干扰项。

### SFT LoRA
- router 池 18 条 → 54 步 ×3 epoch；force_sft 池 199 条 → 597 步 ×3 epoch（项目里最大的池子）。
- merge 校验（相对变化）：

  | LoRA | `[target]` 语言模型层 | `[untouched]` vision tower |
  |---|---|---|
  | router（18 条） | 3.51e-03 | 0.00e+00 |
  | force_sft（199 条） | 7.95e-03 | 0.00e+00 |

  改动幅度与池子大小同向，vision tower 两次都逐位未变。

### 驱动脚本
- `scripts/run_textcraft_full_grid.sh`：A（三条建池臂并行）→ B/C（两次 LoRA + 改键 + merge）
  → D（**十个 eval**，5 配置 × test/deep）→ 双表报告。一次跑完，rc=0，merged 模型用完即删。
- 这次在开跑前加了一道检查：把驱动传的每个 flag 都与目标脚本的 `add_argument` 比对。
  BabyAI 那轮两次失败都是驱动传了脚本不接受的参数（`--domain`、漏 `--experiment-name`），
  而且**失败不报错、只是让下游少做事**。本轮无此类问题。
- GPU 2、3 属于另一位用户，全程未触碰。

## 待办

1. **前三个域的「训练集内」小节需要补池内/池外拆分**（`alfworld` / `scienceworld` / `webshop`）。
   BabyAI 的数据显示聚合 pass_rate 会把「背下训练题」读成「泛化变强」。本域尚未做训练集内重跑。
2. **`babyai_summary.md` 结论 2 要加边界条件**（见本文结论 2）。
3. `train_and_merge` 用 `config.json` 存在与否判断 merge 是否完成，但 `config.json` 在 shard
   写完前就会落盘——中断重跑时可能跳过一个不完整的 merge。本轮是同一进程连续跑完，未受影响。
