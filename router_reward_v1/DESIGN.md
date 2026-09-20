# Router reward 设计文档（v1，暂不执行）

状态：**设计讨论稿，未开工**。本文档只记录方案，不包含任何待运行的实现；下一步动作在文末列出，且明确标记为"暂缓"。

## 1. 问题

`writer_rubrics.py` 里的 `a0_always_both` / `a1_outcome` / `a2_counterfactual` / `a3_budgeted`
四条分配 rubric，是四段人写的英文 prompt，用来决定一条轨迹该不该写成 memory、SFT 数据、两者、或都不写。
现有做法是把整个 dev 集在四条 rubric 下各跑一遍，比较聚合分数选最好的一条。这本质上是在四个离散点上做
prompt 选型，不是在训练一个策略：没有梯度、无法插值或组合、更换 benchmark 或 domain 时要重新手写 rubric。

目标是把"什么样的轨迹该路由到哪里"这件事从人写的 rubric 文本，内化成一个可学习的小型 router 的参数
θ：输入轨迹（及其 base_agent 表现、当前 bank 状态、budget 状态等特征），输出对 `route` ×
`memory_operation` 的一个分布或打分。

`alloc_bank_builder.run_chain` 里 bank 状态是同一 domain 内按任务顺序链式累积的——后面任务的路由要看到
前面任务写进 bank 的记忆——所以这是一个短 horizon（一个 domain 几十步）、小动作空间（route ∈
{memory, sft, both, neither} × memory_operation ∈ {add, refine, replace, null}）的序贯决策问题，
类比训练 tool-call policy 是恰当的。

核心卡点：**训练需要一个 reward，而 reward 目前不存在。** 本文档只讨论 reward 怎么构建。

## 2. 两个时间尺度的 reward，而不是单一 reward

结论先行：不存在一个"又密又准又便宜"的 reward。密集信号靠代理（proxy）产生，用于提供逐样本梯度；
稀疏但无争议的信号靠端到端 held-out 正确率产生，用于验证代理信号是否指向真实目标、以及做低频的策略比较。
两者不是二选一，是互相校准的两层。

### 2.1 稀疏层：held-out 正确率，作为验证信号

**做法**：单独切一个 held-out 任务集，只用来测正确率，不用来生成训练轨迹。每次 router 更新若干版本后，
拿新 router 在 held-out 集上跑一遍（决定路由 → 建 bank/SFT 数据 → 在 held-out 上跑完整 rollout），记录
pass rate。

**必须用同一批次内的均值做基准，不能用历史最高值做基准。** 用户最初提议的"这次正确率 − 上一次
rollout 最高的正确率"会有系统性偏差：G 个带噪样本里取 max 是有偏估计，期望天然高于真实均值，偏差随 G
增大。后果是：只要某一轮里有一个 router 纯粹运气好多蒙对几题，就会成为此后所有轮次的基准，之后的
advantage 全部变成负的，梯度方向变成"压低所有动作"，训练会锁死且不可逆——因为基准只升不降，没有回退
机制。改成同一轮 G 个候选 router 的均值做基准（标准 GRPO 的做法），这个问题就不存在了；"跟历史最优比"
应该只用于挑选最终 checkpoint（best-of-N 语义），不能进 advantage 计算。

**成本和分辨率**（数字来自 [`appworld_experiment/noise_serial_v1/REPORT.md`](../appworld_experiment/noise_serial_v1/REPORT.md) 已验证的确定性配置）：
- 确定性串行配置下，57 条任务一次完整 rollout ≈ 3.5 小时（14 tok/s，非并发，避免 vLLM
  batch-非确定性带来的翻转噪声——并发配置下同一 benchmark 测得过 28.1% 的虚假翻转率，纯粹是 serving
  层的 artifact，不是任务或环境本身的随机性）。
- 57 条题，1 条 = 1.75 个百分点。用配对比较（McNemar，而不是比较两次的总分）而不是比较汇总正确率，
  能省掉大量方差。即便如此，粗估要让"这个 router 比那个好"在统计上站得住，大概需要净赢 4~5 条题以上；
  更小的差异这个信号读不出来，是个粗糙的阶梯函数。
- 想用并发把单次 rollout 时间从 3.5 小时压到几分钟，会重新引入 28% 量级的翻转噪声，而 router 带来的真实
  差异大概率远小于这个量级——两者互相矛盾，不能既要快又要准。**这是本方案最难绕开的成本约束**，只能靠
  "低频验证 + 密集层出梯度"来缓解，无法消除。

**结论**：held-out 正确率因为可信但极贵极稀疏，只适合做验证和最终选型，不适合直接当训练时每一步的梯度
来源。而且它是"一整批路由决策 → 一个标量"的映射（一次 rollout 里 router 对约 90 条训练轨迹各做一次
决策，只换回一个 held-out 分数），把这一个标量的 advantage 摊给 90 个决策，credit assignment 很粗，
样本效率低。这两条决定了它必须是外层的低频信号，不能是内层梯度。

### 2.2 密集层：logprob 差值代理 reward，作为训练梯度来源

这一层解决"每条候选动作都要有独立、低噪声、不用真跑 rollout 的分数"这个需求。

**先固定"探针任务"（probe set）**：从已有的、已验证成功的完整任务轨迹（比如
`noise_serial_v1/run_a` 里 pass 掉 checker 的那些 episode）里，按 domain 分组。给定正在考虑路由的
轨迹 t，探针集合是同 domain 下除 t 本身之外的成功轨迹——不能和 t 重叠，否则是在用一条记忆解释它自己
产生的题，测的不是泛化。

**memory 动作的 reward，具体计算**：

1. 候选记忆内容 M（由 writer 已经生成好的一段文本）。
2. 对探针集合里每个任务 p，构造两个 prompt，除了有没有插入 M 之外完全一致：
   - `prompt_p`：p 原本的 system prompt + 工具定义 + 用户轮次，不带任何记忆；
   - `prompt_p ⊕ M`：同样的东西，只是把 M 插进记忆区，**必须复用 `alloc_writer_harness.render_bank`
     现有的渲染逻辑**，不能另写一套，否则训练时的输入分布和真实推理时不一致。
3. 把探针任务 p 那条已验证成功的轨迹的助手轮次原文，当作固定的目标 token 序列（不用模型重新生成）。
4. 分别对 `prompt_p` 和 `prompt_p ⊕ M` 后接这段固定目标文本，一次前向拿 `prompt_logprobs`（vLLM
   打分接口原生支持，不采样、不解码），把目标 token 段的 logP 加总，得到两个标量。
5. `reward_memory(M, p) = Σ logP(target | prompt_p ⊕ M) − Σ logP(target | prompt_p)`

含义：这条记忆存在时，模型是否更容易"自然而然"说出已验证正确的话。差值 > 0 说明 M 有正贡献，< 0 说明
M 在拖后腿（内容跑偏、误导、或与探针任务无关）。整个过程没有采样，只有两次前向打分，噪声极低。

对整个 domain 的 reward：在探针集合上取平均（可按 M 的 scope 与探针任务的语义相似度加权，避免被完全
不相关的探针任务稀释），再减去与 bank 中已有条目的重复度惩罚（内容与已有条目高度重叠时边际收益本该
打折）。

**neither 动作**：reward 直接记 0。

**both 动作**：目前先用 memory 分量与 SFT 分量相加作为起点，不假设线性可加已经被验证——比如已有一条很
强记忆时同一轨迹的 SFT 增量可能远小于其单独存在时的增量，这类交互项等有数据了再建模。

**SFT 动作的 reward，比 memory 侧贵得多，两个候选方案，都未选定**：

- 方案一（更真实但更贵）：对候选 SFT 样本做几步 LoRA 梯度更新（1~3 步，单样本或小 batch），更新后用
  同一套探针集合重新打一遍 logprob，算训练前后的差值；同时必须测一个域外探针集合（别的 domain）在
  微调前后的 logprob 变化作为遗忘惩罚——如果域外似然掉了，说明这条 SFT 样本在教会一项技能的同时破坏了
  别的能力，要扣分。需要一套"单样本几步微调 → 重新打分 → 回滚权重"的循环，且要固定种子、关 dropout、
  full-batch 单样本梯度以保证这一层本身也是低噪声的。
- 方案二（便宜但是近似）：不真的做微调，用候选 SFT 样本的损失梯度与探针任务损失梯度的点积近似训练
  效果（influence function / TracIn 一类方法的核心公式）。只需要反向传播，不需要真的更新权重再重新
  前向，但直接算全参数梯度点积在数十亿参数模型上开销仍然不小，通常要投影到最后几层或用低秩近似才现实，
  近似质量目前未知。

这部分工程量最大、最容易做错，**明确排在 memory 侧代理 reward 验证通过之后再决定选哪个方案**。

## 3. Router 训练方式

因为代理 reward 对任意候选动作都能独立算，不依赖"实际选了哪个动作才能看到反馈"——同一条轨迹 t 的四个
候选动作（memory / sft / both / neither）的 reward 可以全部提前算出来，不是标准 bandit 那种只能观测到
被选中动作反馈的场景。这意味着 router 训练不需要上 IPS / doubly-robust 那类离线策略优化，可以直接做
监督学习：训练样本是 (t 的特征, 四个动作各自的 reward)，router 学习回归这四个值，或者直接学一个
softmax 去逼近 argmax 动作——标准的多分类/回归问题。

t 的特征来源：现有 `a1_outcome`、`a2_counterfactual` 两条 rubric 里已经写好的判断逻辑（是否 clean
success、base_agent 是否已经会、是 knowledge gap 还是 procedure gap），可以从"最终裁决"降级为
"特征提取器"，连同 `alloc_writer_harness.topic_overlap` 算出的 bank 内 topic-overlap、剩余 budget
比例，拼成 router 的输入特征——这样已经投入的 rubric 设计不会浪费，变成特征而不是被丢弃。

held-out 正确率信号（§2.1）的角色：不直接进入这一步的监督学习梯度（它是一整批决策对一个标量，
credit assignment 太粗），而是作为外层校准——低频跑一次，检查"代理 reward 更高的 router 版本，是否
在 held-out 上也确实更高"。如果不一致，说明代理 reward 被 hack 了或者定义有问题，那时候回来修代理
reward 的定义，而不是直接拿 held-out 分数去算梯度。用户提出的 GRPO + 组内均值基准的思路，保留作为
"如果代理 reward 验证发现系统性跟 held-out 不一致，需要直接对 held-out 信号做策略梯度"时的备用方案，
到时候基准必须是同批次候选 router 的均值，不能是历史最高值（原因见 §2.1）。

## 4. 尚未解决/需要留意的风险

- **代理 reward 是否可信，完全没验证过。** 在写任何训练代码之前，必须先在一个 domain 上，用几十条
  真实轨迹算一遍 memory 侧代理 reward，跟"真的把这条记忆塞进 bank 后跑一次 rollout 看 pass/fail 变化"
  做方向一致性对比。这是唯一的开工前置条件。
- **SFT 侧 reward 两个方案都未选定**，方案一贵、方案二近似质量未知，都需要小规模验证。
- **序贯依赖会让离线复用的决策数据过时**：bank 状态随 domain 内任务顺序累积，如果用旧 router 跑出的
  轨迹序列去训练新 router，新 router 在早期任务上的决策会改变后续任务看到的 bank 内容，训练数据的
  "状态"和线上推理时的状态会逐渐偏离（off-policy drift）。目前没有方案处理，先在短 horizon、小规模
  上看这个偏差是否显著,再决定要不要上重采样或者迭代式数据收集。
- **确定性 serving 消除了 estimator 方差，但不提供"结果对不相关 prompt 扰动有多敏感"的误差棒**——
  之前在 `appworld_experiment/EXPERIMENT_REPORT.md` 等报告的勘误里已经记录过这一点，仍然适用：任何
  用 held-out 正确率做比较时，都需要一个内容无关的 placebo bank 做基线，而不是假设两次跑分的差异全部
  来自 router 的真实差异。

## 5. 下一步（暂缓，不执行）

第一步、也是唯一的开工前置条件：在一个 domain 上跑通 §2.2 的 memory 侧代理 reward 计算（几十次前向
打分，不需要采样、不需要微调、不需要 router 训练代码），跟真实 rollout 结果做方向一致性验证。此步骤
通过之后再决定：(a) 是否值得投入 SFT 侧 reward 的方案一/二；(b) router 用什么参数化（logistic
regression 还是小 MLP）；(c) held-out 校准的具体节奏。

按当前约定，本文档写完即止，以上步骤均不在本次执行。

## 6. 试跑结果（2026-09-03 追加）：§2.2 的原始定义未通过一致性验证

代码：[`src/trajectory_memory_lab/logprob_scoring.py`](../src/trajectory_memory_lab/logprob_scoring.py)（`turn_spans`
+ `score_assistant_turns`，teacher-forcing 打分原语）、
[`scripts/score_memory_proxy_reward.py`](../scripts/score_memory_proxy_reward.py)（驱动脚本）。数据来自
`appworld_experiment/base_train_v2`（90 条 train 轨迹）与已生成的
`appworld_experiment/alloc_banks_v1`（a1_outcome/a2_counterfactual/a3_budgeted 三个 arm 的真实写入记忆）。
结果：[`router_reward_v1/proxy_reward_v1_results.json`](proxy_reward_v1_results.json)。

**结果：38 对 (M, probe) 里，真实同组记忆的 reward 只在 1 对（2.6%）上高于跟随机不相关记忆替换后的
placebo reward；真实记忆的平均 reward 是 −37.95，placebo 平均是 −5.91。** 方向是反的，且幅度大、38 对里
几乎一致，不是噪声。

**诊断（逐 turn 分解，见下例）**：抽查 `a1_outcome_appworld_001`（"部分同事已经在 Venmo 上还过饭钱，剩下
的人发付款请求"）在同组 probe `22cc237_1`（同一类任务的另一次成功尝试）上的分数，插入的记忆内容和这条
probe 轨迹实际执行的计划在语义上是高度一致的——probe 自己在第 5 轮就写道"1. 查 simple_note 里的分账
2. 查 venmo 看谁已经付过 3. 给没付的人发请求"，跟记忆建议的做法基本是一回事。但插入记忆后，
**几乎每一个 assistant turn 的 teacher-forced logprob 都小幅下降**（多数 −0.5 到 −7 nats，24 个 turn 里
只有几个持平或微涨），24 轮累加成 −33.5 的总差值。原因：这条 probe 轨迹是一个*没有记忆*的模型一步步
摸索着写出来的——先列出所有 app、试错式地发现要传 access_token、边探索边用自然语言旁白解释自己在干什么。
一旦模型在 prompt 里已经拿到记忆给出的现成结论，它自己会选择跳过这些探索性旁白和试错步骤，直接执行——
这是记忆"生效"的正常表现，但也正因为它生效了，模型自己的下一步就不再是这段被记录下来的探索式文字，
teacher-forcing 对着这段旧文字打分自然掉分。真实记忆比不相关的 placebo 掉得更多，恰恰是因为它更贴题、
更能真的改变模型的行为，而 placebo 记忆无关痛痒，模型基本无视它，原有文字掉分小。

**结论：§2.2 里"对着一条已验证成功的固定目标轨迹做 teacher forcing"这个定义，在 AppWorld 这种自由格式
推理+写代码、允许多种有效路径的 benchmark 上是系统性反向的——它把"记忆真的起作用、让模型换了一种更好的
走法"和"记忆没用、白白拉低了原路径的似然"混在一起，且前者的信号比后者更强，导致符号整体翻转。** 这不是
实现 bug（`turn_spans` 的边界切分经过校验；idx 对齐修正后逐 turn 数据清晰指向上述机制），是 §2.2 原始
reward 定义本身的问题，v1 的 memory 侧 proxy reward 不能直接用来训练 router。

**没有做的事**：没有因为这次否定结果就去尝试各种修补（比如只打分代码块、只打分决策点、加权早期 turn
等）——按 §3 的自我校准协议，先把结论如实记录，下一步怎么改由使用者决定，不要在同一轮里连续换了三种
定义各跑一遍去挑一个看起来对的。候选修法（都未验证，仅供讨论）：
1. 把打分范围从"整条轨迹的每个 assistant turn"收窄成"记忆声称要触发的那个具体决策点"（比如只打分
   probe 里第一次调用记忆推荐的那个 API 的那一小段），而不是要求复现整条探索式旁白；
2. 换一种不依赖"复现旧路径"的 ground truth：比如用一条*失败*轨迹作为反例目标，reward 定义成"插入记忆
   后，failure 那几步动作的 logprob 有没有下降"，而不是"success 轨迹有没有原样重现"；
3. 放弃 teacher-forcing 代理，直接用 §2.1 的 held-out 正确率做训练信号，接受它贵、稀疏的代价。

三条都没有实现，需要下一步明确要往哪个方向修再动手。

## 7. 决策（2026-09-04）：放弃 §2.2 代理 reward，只用 §2.1 held-out 信号训练

选了候选修法 3。理由：1、2 两条都是在同一个"复现固定目标轨迹"的框架内打补丁，§6 的诊断说明问题不是
"打分范围选错了"或"目标选错了"，而是这类自由格式、多条有效路径的任务本身让"和旧轨迹的文字对得上"这件事
和"记忆是否真的有用"经常反着走——补丁能不能把这个结构性问题压下去,没有把握,验证补丁本身又要再花一轮
类似的实验成本。既然要额外验证，不如直接验证唯一无争议的信号。

**这不是"改小一个方案"，是把训练方式整个换了一条路**，§3、§5 里原来假设"代理 reward 对四个候选动作都能
独立算，所以可以退化成监督学习"的前提不再成立。放弃 §2.2 之后：

- 唯一的 reward 来源变回 §2.1 的 held-out pass rate，而它是"一次 rollout（router 对 domain 内约 90
  条训练轨迹各做一次路由决策）→ 一个标量"的映射，没有独立的逐动作分数了。
- 这就是标准的稀疏终局奖励 RL，不再能绕开 credit assignment：一次 rollout 里的 90 个动作，训练时全部
  共享同一个 advantage（reward − 组内均值），用 REINFORCE/GRPO 的方式对每个被选中的动作做
  `advantage × ∇log π_θ(action)` 的梯度更新。这是无偏但高方差的标准做法，不是新问题，只是之前想靠密集
  层绕开，现在绕不开了。
- §3 里"held-out 只做外层校准，不进梯度"的角色作废，改成 held-out 直接就是梯度来源；§3 原本作为备用方案
  写的"GRPO + 组内均值基准"（§1 里已经把用户最初提议的"跟历史最高比"纠正为"跟同批次均值比"）现在是唯一
  方案，直接采用，基准仍然是同一迭代内 G 个候选 router（或同一 router 的 G 次 rollout）的均值，不用历史
  最高值。
- 成本约束（§2.1 已给出的数字）现在是唯一路径的成本，不再有密集层分摊：一次 held-out 评估 ≈3.5 小时
  （确定性串行，57 题），一次 GRPO 迭代需要 G 个候选各跑一次 held-out 评估（还没算 domain 内训练集上
  建 bank/SFT 数据本身的开销，这部分不需要新 rollout，只是把已有 base 轨迹跑一遍路由决策，相对便宜）。
  G×3.5 小时串行是这条路径上第一个要卡住的实际约束——**没有讨论过 G 取多大、迭代多少轮、串行时间预算
  是否可接受，这是动手前必须先定的数字，不是本文档能替使用者决定的**。

**未决问题（原 §4 风险清单仍然全部适用，额外补充）**：
- G（每次 GRPO 迭代的候选 router / rollout 数）和迭代预算，直接决定这条路径是否现实可行，需要使用者
  拍板一个可接受的总串行小时数，再倒推 G 和迭代轮数。
- router 的参数化（§3 提到的 logistic regression / 小 MLP）现在要直接承担策略梯度更新，需要确认输出的
  是一个可微的分布（对 route × memory_operation 的联合分布做 softmax），而不是像§3 原计划那样先退化成
  回归多个 reward 值。
- 训练集内部序贯依赖（原 §4 第三条）在纯 RL 下影响更直接：同一 rollout 内后面任务看到的 bank 状态由前面
  任务的路由决策决定，一次 rollout 的 90 个决策之间不是独立同分布，advantage 共享是当前唯一处理方式，
  没有验证过是否足够。

本节只记录决策和其直接推论，不包含任何新代码或新实验；G 和迭代预算未定之前不开工。

## 10. Pilot 发现：v1 只有 `add` 是具体的 bug 根因，已修（2026-09-12）

背景：为了在换 reward 方案前先低成本验证"便宜信号有没有用"，做了一个 pilot
（`scripts/pilot_inbatch_reward.py`，`router_reward_v1/pilot_inbatch_v1/`）：取 10 条训练任务为一个
batch，用 `router_iter3.pt` 采样 4 种不同的路由实现（k=0..3），每种分别测（a）**self reward**——用这批
任务自己写出的记忆,回头测这批任务自己的正确率变化;（b）**probe reward**——同一个 bank 测另外 10 道完全
不相关、没贡献过任何记忆的题的正确率变化。两者都相对各自的空 bank 基线算差值。

结果：四个 k 的 self/probe reward 全部是负的，且 **correlation(self, probe) = 0.905**（n=4，量级还很小,
仅供参考）——说明这套便宜信号方向上没有跟真正关心的泛化效果脱节,值得继续探索;但同时也发现四次全负,
说明当前这版 router 检查点本身路由质量有问题。

**具体查了"为什么"**：对比 active_entries 最多的 k0（8 条）和最少的 k3（3 条）的 bank 内容和检索记录，
发现 k0 里 000/001/002 三条、003/004 两条、005/006 两条分别是同一件事的近乎重复表述——因为这批任务里
`07b42fd_*`/`229360a_*`/`22cc237_*` 各是同一场景的 3 个变体，router 对每个变体独立采样,v1 设计里
`memory_operation` 只有 `add`、没有 `refine`/`replace`,于是同一条结论被反复"新增"成 2-3 条近似重复的
记忆。逐任务比对检索记录证实了后果：`229360a` 这组在 k0 下检索到 `[003,004,007]`——003/004 是重复内容,
007 是完全不相关的话题（歌曲属性查询混进了库管理场景的检索结果）,top-3 里 2 个位置被重复占用、1 个位置
被无关内容占用,这组任务 3/3 全部失败；k3 只有 3 条记忆,检索到更概括、覆盖面更广的单条记忆,同一组任务
2/3 成功。（但也验证了不是"条数越少越好"这么简单——`22cc237` 这组反而是 k0 更好,真正起决定作用的是
"检索槽位有没有被无关/重复内容占用、命中的那条内容对不对题"，条数只是偶然相关。）

**修法**：不需要让 router 学"选哪条旧记忆去 refine"——这正是 v1 当初刻意回避的变长动作空间问题。判断
"新内容是不是已经有近似的旧记忆"完全不需要学习，用内容相似度就能确定性判断,而这个判断逻辑
（`alloc_writer_harness.topic_overlap` + `REFINE_TOPIC_OVERLAP_MIN` 阈值）在旧的 LLM rubric 系统里
已经写好、验证过，只是原来只用来"校验 LLM 自己说的 refine 靠不靠谱"。现在反过来用：内容生成完之后
（`router_bank_builder._dedup_against_active_bank`），无论 router/写手说的是什么 operation，只要跟
bank 里现有活跃条目的相似度超过阈值，一律强制转成 `refine` 并指向那条最相似的条目（原条目标记
`superseded`），router 的参数化、特征、训练方式完全不变，只学 route 这一个 4 分类。用合成的近似重复
内容做了冒烟测试：4 次近乎重复的写入尝试正确合并成 1 条活跃记忆（前 3 次被顺序 supersede），而不是堆出
4 条重复条目。

**还没做的事**：没有拿这个修复重新跑一遍 pilot 验证 self/probe reward 是否真的改善——这是修复"对不对"
的直接验证，成本和上次 pilot 相当（~5 小时），值不值得现在就跑，留给下一步决定。

## 11. 并行 serving：拆成两个 TP=2 副本（2026-09-12）

动机：确定性 serving 靠 `--max-num-seqs 1` 保证同一个 server 任意时刻只处理一个序列——这意味着对**同一个**
server 加客户端并发完全没用，服务端还是会把请求排成队串行处理，总耗时不会缩短。真正能加速的只有起
**多个独立的 server 副本**，让它们各自内部继续保持"单序列串行"（determinism 不受影响），但副本之间
互相并行。

**可行性先验证了一下**：Qwen3.5-35B-A3B 是混合线性注意力架构（`layer_types` 里 4 层里只有 1 层是
`full_attention`，其余是 `linear_attention`，只有 full_attention 层需要随序列长度增长的 KV cache），
模型权重本身 67GB bf16。原来 TP=4 时每卡只用了 44GB 里的一小部分给权重（17GB 左右），大头是
`--gpu-memory-utilization` 预留的 KV cache 池。算下来 TP=2（每卡 33.5GB 权重）大概率能在 49GB 单卡显存
里放得下,试了一下确实可以：两个 TP=2 副本（GPU 0,1 和 GPU 2,3，`--gpu-memory-utilization 0.85`）都
干净启动，各用 42GB，没有 OOM。

**确定性也重新验证了**（TP=2 是没验证过的新配置，不能想当然复用 TP=4 的结论）：同一个 prompt 对同一个
副本连发 3 次,输出逐字节相同;两个副本之间对同一个 prompt 也逐字节相同。这只是一个短 prompt 的轻量抽查,
不是 `noise_serial_v1` 那种全量 57 题级别的验证,但作为"先用起来、边用边看有没有异常"的门槛足够。

**用法**：`scripts/pilot_inbatch_reward.py` 现在把 self-eval 和 probe-eval（原本顺序跑）改成用
`launch_subset_eval`/`wait_subset_eval` 拆成"发起-等待"两步，分别指向 `--base-url`（8000,顺带也用来做
建 bank 阶段的内容生成）和 `--base-url-probe`（8001），两个真正并发跑，单次 pilot 的墙钟时间从 ~5h
压到 ~2.5h。后续如果要跑真实训练循环（而不只是一次性 pilot），这套"多副本"思路可以推广成更大规模的
任务调度（比如 GRPO 一个 iteration 内 G 个 rollout 分摊到多个副本上跑），但目前只在这一个 pilot 脚本
里落地，`train_router_grpo.py` 还没改。

## 8. v1 实现范围收窄（2026-09-04 开工时追加）

预算：G=4，先跑 1 轮迭代，看链路通不通再决定要不要继续（原设想上限 3 轮 ≈ 42 小时，不锁死）。

代码：`src/trajectory_memory_lab/router_policy.py`（router 本体，线性层，5 个特征 → 4 类
route 的 softmax，~24 个参数）、`src/trajectory_memory_lab/router_bank_builder.py`（`run_chain`
的 router 版）、`src/trajectory_memory_lab/writer_rubrics.py` 新增的 `routed_writer_system`
（route 已经由 router 定好，LLM 只负责按这个 route 把内容写出来，不再自己选 route）、
`scripts/train_router_grpo.py`（GRPO 训练循环）。

两处刻意收窄的范围，写在这里以免结果出来后才被当成 bug：

1. **router 只学 route 这一个 4 分类**（memory/sft/both/neither），不学 `memory_operation`
   里的 refine/replace——命中已有条目去精修是一个变长动作空间（要在当前 bank 的条目里选一个），
   v1 只学 `add`，把它留到 v2。
2. **SFT 这条路径只记录路由决策、参与梯度计算，但不真的构建或应用任何 SFT 产物**（不做 LoRA
   微调）。held-out 评测时 agent 只有 base 模型 + memory bank 检索，没有任何 SFT 权重更新。也
   就是说这一版 held-out reward 里，只有 memory 轴真的在影响分数，SFT 轴的 reward 贡献是"router
   学会了在这类轨迹上说 sft"这件事本身有没有跟 memory 轴形成合理的整体路由,而不是 SFT 数据本身
   有没有用——§2.2 里搁置的 SFT 侧 reward 方案一/二仍然没有解决,这一版只是先不让它挡路。

首次真跑记录：iteration 1，G=4，2026-09-04 17:32 UTC+8 左右启动，确定性 vLLM server（tmux
`det_server_router_grpo`）+ 训练循环（tmux `router_grpo_iter1`，日志
`router_reward_v1/train_iter1.log`）。冒烟测试（3 条真实轨迹、stub 掉 LLM 调用、检查 bank 正确
写出 + 梯度确实能传 + optimizer.step() 后参数真的变化）先于任何 GPU 调用跑过并通过，之后才启动
server 和真实训练。

## 9. Credit assignment 太粗：按 batch 切分 reward checkpoint（2026-09-05）

iteration 1 跑完后发现的问题：一条链上 90 个路由决策共享同一个 reward（链末尾那一次 57 题 held-out
分数），决策 #3 和决策 #87 拿到的 advantage 完全一样；而且一次 iteration（G=4）只有 4 个 reward
观测点，样本效率很低。四个 rollout 的 pass_rate 是 0.3684 / 0.4561 / 0.5263 / 0.4737——单是随机
初始化的 router 采样出的路由差异,就能让最终成绩摆动 16 个百分点,说明这个信号里有相当一部分方差
来自"当前决策组合恰好如何"而不是"router 参数往哪个方向学"，需要更细的 checkpoint 才能把两者分开。

**改法**：把每条链的 90 个任务按顺序切成 B 个连续 batch（bank 状态跨 batch 累积，不重置；router
的"进度"特征——position/task_ids_len——也按全局 90 算，不按单个 batch 重置）。每个 batch 处理完，
在**完整的 57 题 held-out 集**（不缩小子集）上测一次当前累积 bank，reward 记为这次 checkpoint 分数
减上一次 checkpoint 分数（batch 0 的"上一次"直接复用 `noise_serial_v1` 里已经验证过零翻转的空 bank
基线 25/57，不用重测）。advantage 仍按 GRPO 的组内均值算，但现在是"同一个 batch 位置上,四条链比",
不是"整条链跑完再比"。一次 iteration 只做一次反向传播和一次 optimizer step,θ 在整个数据收集过程中
不变,仍然是干净的 on-policy 更新，只是 reward 观测点从 4 个变成 4×B 个,每个观测对应的决策数从 90
降到 90/B。

**为什么没有把 held-out 子集也缩小来省钱**：57 题子集换成比如 15 题,标准误差能到 ±13 个百分点,
比单个 batch 带来的真实边际增量还大,等于用更贵的频率换回同等甚至更差的噪声,不划算。保留完整 57 题,
代价就是总时长跟 B 线性增长。

**预算**：B=3（30/30/30），单次 iteration 总耗时 4×3×3.5h ≈ 42h（约当前设计的 3 倍）。

代码：`router_bank_builder.run_router_chain` 新增 `initial_bank`/`start_position`/`total_task_count`
三个参数，支持把一条链拆成多段调用并保持累积状态和连续的进度特征；返回值新增完整 `bank`（含
superseded 条目）供下一段续用。`train_router_grpo.py` 重写为按 batch 收集 checkpoint、按 batch 位置
算 advantage。跑真实 GPU 之前用 stub 掉 LLM 调用和 held-out 评测的合成测试验证过：bank 跨 batch 正确
累积（active_entries 单调不降）、每条链的决策数之和等于 90、梯度能传、参数会变。

## 12. Route 分布熵坍缩：加熵正则 + 降学习率（2026-09-16，v4）

**现象**（此前几节都没记，补在这里）：`cheap_train_v2` / `cheap_train_v3_g4` 反复出现 route 分布坍缩。
两种表现要分开看，因为它们出现的时间不一样：

- **greedy（argmax）先坍**：v2 iteration 1 的 validation 是 `routes={'both': 90}`——90 个决策全走 `both`，
  `memory`/`sft`/`neither` 一次都没被选中。v3_g4 是 `{'both': 57, 'sft': 33}`，同样只剩两个 route。
- **采样分布后坍**：同一时刻 v2 的采样熵其实还有 ~1.0-1.3 nats（ln4 = 1.386 是上限），并没有真的坍到 0。
  也就是说 argmax 已经完全退化的时候，底下的分布只是"排序稳定"、还没饱和。真正的熵坍缩在 v3_g4 才看得
  清楚：9 个 batch 里 mean entropy 从 1.138 单调掉到 0.653。

**教训**：只看 validation 的 `route_counts` 会晚一步——argmax 是个阶跃函数，排序一稳定它就全变成同一个
route，但那时分布本身还有救。所以 v4 把**分布的熵**也记进日志（训练每个 batch 一条 `mean_entropy`，
validation 也记一条），把它当早期预警，而不是等 `route_counts` 变成单一值才发现。

**根因不止一个，而且主因不是"没有熵正则"**：advantage 的量级很小（实测 ±0.05 ~ ±0.3），但优化器是
Adam——Adam 按梯度的 running second moment 做归一化，梯度再小，参数步长也还是 ~lr 量级。lr=0.05 作用在
一个只有 24 个参数的线性模型上，9 步就足够把 bias 推到 logits 饱和。做了一次受控模拟（把 advantage 建模
成正比于该候选里 `both` 的比例，即真实的系统性压力，9 个 batch = 1 个 iteration，3 个 seed 平均）：

| lr | entropy_coef | H(batch0) | H(batch8) | 变化 | p(both) |
|---|---|---|---|---|---|
| 0.05 | 0 | 1.313 | 1.061 | −0.252 | 0.640 |
| 0.05 | 0.01 | 1.313 | 1.215 | −0.098 | 0.544 |
| 0.05 | 0.03 | 1.313 | 1.344 | +0.031 | 0.370 |
| 0.01 | 0 | 1.313 | 1.321 | +0.008 | 0.283 |
| 0.01 | 0.01 | 1.313 | 1.347 | +0.033 | 0.278 |
| 0.01 | 0.1 | 1.313 | 1.362 | +0.049 | 0.241 |

读法：**lr 从 0.05 降到 0.01 已经把坍缩压力基本消掉**（−0.252 → +0.008），熵正则是在这之上再加一层保险。
选 `entropy_coef=0.01` 而不是更大，是因为 coef=0.1 时 p(both)=0.241 已经贴着均匀分布 0.25——等于把 reward
信号整个淹掉，router 什么也学不到；coef=0.01 下 p(both)=0.278 仍然高于 0.25，说明信号还在起作用。
（注意这是无噪声的系统性信号模拟，真实 advantage 有噪声，方差会更大；这张表只用来定**相对量级**。）

**改动**：
1. `router_policy.py` 新增 `action_distribution()`，`sample_action()` 返回值从 `(route, logprob)` 变成
   `(route, logprob, dist)`——返回整个 `Categorical` 而不只是熵，因为 `.probs` 还要拿来做逐 route 诊断。
   两者都挂在计算图上，所以熵项可微。
2. `router_bank_builder.py` 每条决策额外记 `entropy` / `probs`；greedy 路径也记（`no_grad`），这样
   validation 也能看到分布，而不是只有 argmax。
3. `train_router_selfreward.py` loss 变成 `pg_term - entropy_coef * entropy_sum`，两项都在"所有候选 ×
   所有决策"上求和，量级自然对齐（都随 K × batch_size 增长）。新增 `--entropy-coef`（默认 0.01），
   `--lr` 默认 0.05 → 0.01。日志新增 `pg_term` / `mean_entropy` / `entropy_coef`。
4. 产物目录 `cheap_train_v3` → `cheap_train_v4`，record_protocol 和 experiment 名一并改成 v4，
   避免跟已有 checkpoint / 日志混在一起。

**冒烟测试**（都不占 GPU，跑在真实训练启动之前）：
- `scripts/smoke/smoke_entropy_regularization.py`：验证 `sample_action` 确实返回完整分布、熵可微、
  熵等于 −Σp·log p；构造一个已经确定性的分布（bias=6.0，H=0.052），确认熵项的梯度把它推向更分散
  （50 步后 H=1.148，p(both) 0.993 → 0.576）；以及 entropy_coef 越大分布越散的单调性。
- `scripts/smoke/smoke_batch_update_e2e.py`：只 stub 掉两个边界（writer LLM 的 `ModelClient`、AppWorld
  评测子进程），跑**真实的** `run_one_batch_update`，确认参数确实更新、熵项确实进了 loss
  （`loss != pg_term`，而 `entropy_coef=0` 时两者相等）、同一 seed 下开熵正则比不开熵更高。

**一个校准上的坑，记下来免得下次踩**：从一个**已经完全坍缩**的 router（bias=6.0）出发，lr=0.01 时恢复
非常慢——Adam 每步最多挪 ~lr，抹平 6 个单位的 bias 需要几百步，不是几步。所以熵正则的作用是**预防**，
不是**救回**：一旦 checkpoint 已经坍了，正确做法是重新初始化，而不是指望加了熵正则接着训能自己爬回来。

**本次真实训练**：`--rollouts-per-batch 8 --batch-size 10 --lr 0.01 --entropy-coef 0.01`，
输出 `router_reward_v1/cheap_train_v4/`。两个 TP=2 副本跑在 GPU (4,5) 和 (6,7)，端口 **8010 / 8011**
——不是 §11 的 8000/8001，因为这台机器（COE-CS-sv002）上 8000-8002 被别的用户占着。

**顺带修正 §11 的一个结论**：§11 说"两个副本之间对同一个 prompt 也逐字节相同"。这次在 COE-CS-sv002 上
用 5 个 prompt 重测，**4/5 相同、1/5 不同**（差异出现在第 180 个字符之后，是 `` `song_id` `` vs
`song ID` 这种措辞级差别，典型的 MoE expert GEMM atomics 累加顺序不确定性）。副本**各自内部**仍然是
确定性的（A 连发 3 次、B 连发 2 次都逐字节相同），所以 §11 "副本内确定"的部分成立，"副本间确定"那半句
是拿单个短 prompt 抽查得出的，过强了。

对本实验的影响：self-eval 按 k 的奇偶分到两个副本上，所以同一批候选之间的 pass_rate 差异里，混进了一点
"跑在哪个副本上"的噪声（rollout 有 25-77 步，一个 token 分岔会被放大）。但副本分配（k 的奇偶）与采样出的
route 是相互独立的——route 由 `torch.manual_seed(...+k)` 决定，跟副本无关——所以这是**方差**，不是
系统性偏向某个 route 的**偏差**，不会伪造出"`both` 更好"的信号。v2/v3 用的是同一套机制，所以这不是 v4
引入的新问题。留作已知项：如果以后要压这部分方差，办法是让同一个候选的 self-eval 固定跑在同一个副本上、
或者干脆同副本串行（代价是墙钟时间翻倍）。

## 14. Reward 组内比较丢掉了无记忆基线；router 特征改造；SFT 真正训练进去（2026-09-18）

三件相关但独立的改动，都是同一天做的，写在一起：

### 14.1 GRPO advantage 把无记忆基线抵消掉了

`train_router_selfreward.py` 一直是这样算的：
```python
rewards = [c.self_pass_rate - self_baseline for c in candidates]     # 跟基线比
advantages = [r - mean(rewards) for r in rewards]                     # 再减组内均值
```
代入化简：`advantage_k = (pass_k - baseline) - mean_j(pass_j - baseline) = pass_k - mean_j(pass_j)`——`self_baseline` 精确抵消，从未真正进入梯度。router 只学到"比同批候选好/差"，即使这一整批候选全都不如什么都不写，也学不到"不写"更好。这与本项目至今每一次 validation（v2=0.3684、v3_g4=0.4035）都低于基线（重新核对前的 0.4386）这个现象一致。

**改法**：把 `self_baseline` 当成组里免费的第 K+1 个参照点一起取均值（它是已知量，不需要额外评测）：
```python
group_mean = mean(pass_rates + [self_baseline])
advantages = [p - group_mean for p in pass_rates]
```
这样如果一批候选普遍不如基线，advantage 会整体转负。

### 14.2 副本不可比：0.4386 这个基线本身需要重新验证

另一台机器同一天在 §13 里证明了跨副本/跨运行的绝对分数不可比（差距可达 10-20pp）。核查发现 0.4386 来自 `noise_serial_v1/run_a`，生成于 2026-08-29——比两副本确定性 serving 搭建（09-12）还早，几乎肯定是不同服务端配置。G=8 那次"超过基线"的结论（0.5789 vs 0.4386）因此不成立，已经在 G=8 用的同一个副本（127.0.0.1:8000）上补跑一次匹配的无记忆基线（`router_reward_v1/baseline_recheck/`）。

### 14.3 Router 特征：去掉两个线性冗余特征，换成真实内容的哈希词袋

`frac_remaining = 1 - frac_position`，给定 `total_task_count` 恒为 90，对线性模型是精确的线性冗余，零边际信息量；`frac_position` 本身也一直没有证据表明有用。两个都删。换成两块哈希词袋特征（`router_policy.TEXT_HASH_DIM=16`，md5(token)%16 计数后归一化，不训练、不是 embedding 模型）：
- 最近两个 batch 里 bank 新增/变动条目的实际文本（`router_bank_builder._recent_changes_text`，靠 `alloc_writer_harness.apply_memory_operation` 新增的 `created_position` 字段回溯）；
- 这道题已经起草好的候选内容的实际文本（`_draft_content_text`）。

配合的架构改动（"content-before-route"）：`router_bank_builder.run_router_chain` 现在**先起草内容再决定路由**，而不是先路由再让 LLM 照着写——router 看到的是真实草稿文本的哈希特征，不是数字代理。代价：每道题都要起草一次内容，即使最后路由是 `neither`（以前 `neither` 完全不调 LLM）。`FEATURE_DIM` 从 5 变成 `3 + 2*16 = 35`。

### 14.4 SFT 真正训练进去，写手换成专门的 SFT writer

此前 `sft_plan` 只是记录一条 `repair_target` 一句话描述，从未被消费——route=`sft` 和 route=`neither` 因此在 reward 上完全等价（都不改变任何评测结果），router 根本没有机会学会二者的取舍。

**内容生成换人**：memory 和 sft 不再共用一次"both"起草调用。memory 仍走 `routed_writer_system("memory")`；sft 换成新写的专用 writer（`appworld_sft_writer.APPWORLD_SFT_WRITER_SYSTEM`），只在 `base_agent_success=False` 时调用（成功的题没有"错误"可修）。这个 writer **不是另外训练的教师模型**，是同一个基座模型换一个 system prompt——参考 tau2-bench 那条已经跑通的"SFT-data writer"经验（`tau_sft_data_writer.py`），但产出形式不同：tau2-bench 那边产出一段可以逐字节脚本回放的 `assistant_turns`（因为要喂给一个模拟用户参与的对话回放框架）；AppWorld 没有模拟用户轮次、是纯代码执行循环，逐字节脚本回放一碰到任何一个没预测到的真实 API 返回值就会脱轨，所以这里让 writer 只产出一段**自然语言修复计划**（具体该调哪些 API、顺序、原来错在哪），像 `memory_block` 一样注入到一次全新尝试的初始 user message 里（`appworld_agent.build_initial_user_message` 本来就是纯文本拼接，零改动可以直接复用），交给一个真实的 agent 在真实环境里重新执行、自己应对真实返回值。

**只有真正 replay 成功的才算数**：`scripts/run_appworld_guided_replay.py` 对 committed 的 sft/both 决策，用这段计划文本重新跑一次全新的这道题；`trajectory_memory_lab.router_sft_pipeline.replay_and_verify` 只在 AppWorld 自己判定 `success=True` 时才把这次 replay 的**真实对话记录**（不是 writer 的计划文本本身）转成一条训练样本——计划只是提示，标签永远来自真实环境的真实反馈。

**只 replay 被选中候选的 sft 决策**：一个 batch 有 K 个候选，只对 `random.choice` 选中、真正延续到下一 batch 的那个候选做 replay，不是全部 K 个——否则 AppWorld 评测成本乘以 K，而其余 K-1 个候选的 bank 反正不会被继续使用。

**训练触发**：验证样本积累到 `TRAIN_TRIGGER_SIZE=8`（跨过一个 8 的倍数即触发一次，不是每条都触发）就跑一次 `scripts/router_sft_lora_update.sh`：
1. 暂停 `det_server_b`，空出 GPU 2、3，用 `.train-venv/bin/swift sft`（ms-swift，LoRA rank 8，复用仓库里 `train_qwen_lora.sh` 已验证过的超参）在整个累积池（不是增量续训，每次都从 base model 全量重训，避免多次小步 LoRA 叠加的漂移风险）上训一次；
2. 暂停 `det_server_a`，用 `scripts/merge_qwen_lora.py`（CPU-only，不占 GPU）把 LoRA 合并进 base weights，产出一份独立的合并 checkpoint；
3. **两个副本都**指向这份合并后的 checkpoint重新拉起，`served-model-name` 仍是 `qwen35-tau`，端口/TP/GPU 配置和原来完全一致。

**为什么两个副本都要重载，而不是像 `serve_tau_agent_sft_lora.sh` 那样只给一个副本挂 `--lora-modules`**：det_server_a/b 至今被当成完全等价的两个副本，候选按 k 奇偶分派纯粹是为了并行、和路由采样无关（§11/§13 的论证基础）。如果只有一个副本换成微调后的模型，"candidate 落在哪个副本上"会突然变成一个决定它能不能看到微调效果的、系统性的因素——这正是 §13 花一整节讲清楚的"副本不等价"陷阱，会直接在 reward 里种下一个混淆变量。合并权重、两边都重载，是唯一能保住"副本可互换"这个前提的做法。代价是每次触发要多一步 CPU 合并（本机 503GB 内存、4.8TB 硬盘余量，够用）和两个副本的重启时间。

**尚未验证**：这一整条链路（SFT writer → guided replay → 训练触发 → 双副本重载）还没有跑过一次真实端到端；下一次正式训练如果触发了 SFT 训练，第一次触发时需要盯着看 `router_sft_lora_update.sh` 的输出确认两个副本都正常起来了。

## 13. 记忆后端三方对比与检索量实验（2026-09-17，v5/v6）

起因：v4 训练跑到 batch 2 时发现记忆相对无记忆基线大幅掉分，逐条追查后做了一整轮受控对比。
结论先写在前面：**三个后端之间的差别，远小于「用不用记忆」本身的效应；而记忆整体在这个
setup 下平均为负。**

### 13.1 方法学：三个必须先修的对照问题

这轮最大的收获不是某个数字，而是三个对照缺失，每一个都曾让我得出错误结论：

1. **副本之间不等价。** 同一个 bank、同一 seed、同一批任务，TP=2 副本给 0.70、PP=3 副本给
   0.60；两个同为 TP=2 但 GPU 不同的副本也能差 10pp。**跨副本的绝对分数不可比。**
   修法：每个候选的所有实验臂绑定到同一个副本，只比较候选内部的差值。
   代价是必须为每个副本单独跑无记忆锚点。
   三个副本各自的确定性对照（同 bank 同副本重跑）全部通过：k1/k4/k6 逐题一致。

2. **基线用错。** 此前所有「比基线差 X pp」都以 `base_train_v2` 的 0.80 为准，但那是另一次运行、
   另一套服务端记录的。实测三个副本的真实无记忆分数是 **A=0.70、B=0.70、C=0.60**。
   这一项单独就让此前所有损害估计高估了 10–20pp。

3. **mem0 的 `threshold=` 参数不按它返回的 score 过滤。** 实测 `threshold=0.30` 仍返回 0.287 和
   0.231，`threshold=0.50` 保留 0.287 却丢掉 0.231。直接用它会得到「阈值无效」的错误结论。
   改为在 `mem0_store.search_entries` 里按返回分数自己过滤。

同一个教训重复了五次：**任何跨运行、跨服务端、跨配置的比较，都必须先有同条件对照。**

### 13.2 三方对比结果（8 候选 × 3 后端 = 24 次评测，全部同副本）

| 候选 | 副本 | 无记忆 | v4(破坏性dedup) | v5(非破坏性合并) | mem0 | mem0+阈值0.30 |
|---|---|---|---|---|---|---|
| k0 | C | 0.60 | 0.50 | 0.60 | 0.60 | 0.60 |
| k2 | C | 0.60 | 0.50 | 0.50 | 0.50 | 0.60 |
| k3 | A | 0.70 | 0.70 | 0.70 | 0.70 | 0.80 |
| k4 | C | 0.60 | 0.60 | 0.60 | 0.60 | 0.50 |
| k5 | A | 0.70 | 0.70 | 0.70 | 0.30 | 0.60 |
| k6 | B | 0.70 | 0.50 | 0.50 | 0.50 | 0.50 |
| k7 | B | 0.70 | 0.60 | 0.60 | 0.50 | 0.40 |

平均 Δ vs 无记忆（n=7）：**v4 −7.1pp、v5 −5.7pp、mem0 −12.9pp、mem0+阈值 −8.6pp**。
**没有任何一个后端在平均意义上超过「不用记忆」。**

### 13.3 §12 那个 dedup bug 的真实影响：+1.4pp

v4 → v5 平均只有 +1.4pp（−7.1 → −5.7）。bug 是真的、机制也查清了（Jaccard 0.25 阈值把互补
记忆当重复合并，平均只保留旧条目 72% 的内容），但**对最终性能几乎无关紧要**。
修复只改到 8 个候选中的 2 个（k0、k2），其中只有 k0 有 +10pp。

过程中我连续四次高估这个修复（+20pp → +10pp → +5pp → +1.4pp），每次都是因为拿跨副本数字比较。

### 13.4 检索量是比后端更强的变量

在 k1 的库上做的剂量-反应扫描（同副本 C、同 10 题）：

| 实际注入/题 | 配置 | 分数 |
|---|---|---|
| 0.0 | 无记忆 | 6/10 |
| 0.6 | mem0 阈值0.50 | 7/10 |
| 1.1 | mem0 阈值0.30 | **9/10** |
| 1.7 | BM25 top3 | 7/10 |
| 3.0 | mem0 无阈值 | **3/10** |

倒 U 形，峰值在约 1 条/题。无阈值 mem0 的 3 条/题比完全不给记忆还差一半，且终止状态佐证
（4 次耗尽步数 vs 阈值版 10/10 正常完成），说明 agent 确实被低相关度记忆带偏。

关键细节：**`--memory-top-k` 这个参数本身不是变量，实际注入条数才是。** BM25 在 top_k=3/5/10
下实际注入恒为 1.7 条（分数为 0 的不返回），所以扫 top_k 对 BM25 毫无作用；mem0 无过滤、
硬填满 k 条，才真正改变了注入量。`top_k=3` 这个从 tau2-bench 继承的默认值，此前从未被扫过。

### 13.5 阈值复现失败：k1 是离群值

k1 那个 0.90（+30pp）没有推广。7 个候选加阈值后：**平均 −8.6pp，1 好 2 平 4 差**。

分成两个独立结论：
- **阈值确实改善 mem0 自身**：平均 +11pp（k5 从 0.30 救回 0.60，k1 从 0.30 到 0.90）——
  「无过滤硬填满 top-k 有害」这个机制成立。
- **但改善后仍普遍打不过不用记忆**：说明对多数库而言，问题不只在注入量，**内容本身就没有
  正价值**。k6 是最干净的反例：过滤掉噪声后分数纹丝不动（0.50），仍比无记忆低 20pp。

### 13.6 对 router 训练的影响

记忆效应在候选之间的摆动幅度是 **+30pp 到 −40pp**，远大于 router 路由决策所能产生的差异。
这解释了 §12 里那个一直测不出信号的 reward：**被 router 控制不了的变量（记忆内容质量、
检索注入量）主导了 reward 的方差。** 在记忆本身能稳定产生正收益之前，训练 router 去分配
记忆是在优化一个信噪比过低的目标。

### 13.7 代码改动

- `router_bank_builder._dedup_against_active_bank`：强制 refine 改为非破坏性合并
  （保留率 <0.9 时把旧文本并入新条目），§10 的真重复合并行为由冒烟测试守住
- `src/trajectory_memory_lab/mem0_store.py`：mem0 后端，LLM 指向本地确定性 vLLM，
  embedder 用 fastembed（**刻意避开 sentence-transformers**，它会把 transformers 从 4.57.3
  升到 5.17.0 而 vLLM 0.17.1 依赖前者），向量库为每候选独立的本地 qdrant，
  posthog 遥测在 import 前关闭
- `scripts/serve_mem0_retrieval.py`：sidecar。`appworld_venv` 是 pydantic 1.10/SQLAlchemy 1.4，
  mem0 要 pydantic 2/SQLAlchemy 2，直接装会搞坏 AppWorld 评测环境，故走 localhost HTTP
- `memory_retrieval`：bank 路径是目录则走 mem0，是 .json 则走原 BM25，旧结果仍可复现；
  新增 `MEM0_SCORE_THRESHOLD` 环境变量
- `serve_appworld_deterministic.sh`：PORT/TP/PP/显存比例改为环境变量，默认值不变

接 mem0 时踩的四个坑都是**静默失败**（`add` 返回 `{"results": []}` 不报错）：
reasoning parser 吞掉 content、`custom_fact_extraction_prompt` 在 2.0.20 是死配置、
默认抽取 prompt 面向个人助理会丢弃 API 知识、`response_format={"type":"json_object"}`
在本地 vLLM 上产出非法 JSON。sidecar 的检索路径因此特意让错误抛出而非吞成空结果。

### 13.8 下一步的判断

不建议继续在后端/检索层调优——三个后端在 7 个候选里有 4 个给出完全相同的分数，这一层
已经不是瓶颈。真正未解的是 **13.5 的后半句：记忆内容本身没有正价值**。
该做的是检查写手产出的记忆是什么、为什么对解题没用（例如是否只是重复 API 文档里已有的信息），
而不是换第四种存储方案。

## 15. SFT 修复计划的写手换成外部 teacher model，默认开（2026-09-18）

`appworld_sft_writer.py` 现在有两个互相替代的写手，产出格式完全一样
（`validate_writer_output` 的 `{"plan","mistake_summary","evidence_steps"}`），下游（guidance 注入、
guided replay、router 特征哈希）不需要知道也不需要关心是谁写的：

- `self`：跟 task agent 同一个基座模型（qwen35-tau），走本地 `ModelClient`，`APPWORLD_SFT_WRITER_SYSTEM`。
- `teacher`（**新默认**）：外部 Gemini 模型（`generate_plan_with_teacher`，`google-genai` SDK），
  `GEMINI_TEACHER_SYSTEM` 是同一套 branching 逻辑（`success` 字段决定是写"修复计划"还是"提炼已成功路径"）
  的 Gemini 版本。

**为什么默认换成 teacher**：running_log.md §11 的 `probe_sft_repair_yield_gemini_teacher.py` 在 33 道
`base_train_v2` 失败题上测出 Gemini teacher 修复率 15/24 = 62.5%（分母是"control 组无引导重跑仍然失败"
的真正denominator）。同一批题的自写手版本（`probe_sft_repair_yield.py`）**没有跑完**，只有 1 道题完整
跑完 repair 且还是"计划把好使的搞坏了"，不能严格对比，但 teacher 的 62.5% 已经足够作为默认值的依据——
之前 sft 路由完全没训练进去（DESIGN.md §14.4），有个能打的写手比没有强。

**接入方式**：`RouterBuilderConfig` 新增 `sft_writer`（`"teacher"` 默认 / `"self"`）、`teacher_model`
（默认 `gemini-3.1-pro-preview`——`gemini-3-pro-preview` 已经 404，是 API 报错里直接给出的替代型号）、
`teacher_api_key_file`（默认 `/nas04/yixuh/.config/continual-memory/gemini_api_key`）。
`train_router_selfreward.py` 对应加了 `--sft-writer`/`--teacher-model`/`--teacher-api-key-file` 三个 CLI
参数，`router_bank_builder.run_router_chain` 里原来"self writer"那段直接分支成 `if config.sft_writer ==
"teacher": ...else: ...`，两边失败都统一返回 `None`（`writer_output`），后续处理完全不用改。

**已知代价**：teacher 分支是外部网络调用，比本地 vLLM 慢且依赖网络可用性；这台机器上 `import
google.genai` 本身实测过要 2-3 分钟（NFS I/O 慢，不是包本身的问题），只在进程生命周期内发生一次
（后续调用复用已导入的模块），但训练脚本每次重启都会重新付一次这个代价。

## 16. Router 变成可训练的 LLM：4 路 categorical，而不是自由生成（2026-09-18，router_llm_grpo_v1）

把 router 从 144 参数的线性分类器（`router_policy.RouterPolicy`）换成一个 LoRA 可训练的小 LLM，
reward / batch 结构 / advantage / 随机选候选延续 bank 这些全部沿用 `train_router_selfreward.py`，
**只换"谁做决定"和"梯度打在哪"**。`train_router_selfreward.py` 一行没动，是并列实验不是替换。

### 16.1 核心发现：这个动作空间不需要 vLLM，也不需要 LoRA 热加载

交接文档给的方案是 RLHF 的标准三段式：vLLM 采样拿 token logprob → 本地 peft 副本 teacher-forcing
重算可微 logprob → `POST /v1/load_lora_adapter` 把新权重同步回 vLLM。**这两个机制在本仓库的
vLLM 0.17.1 上都确实存在**（已查证，不是假设）：

- `entrypoints/serve/lora/api_router.py`：`POST /v1/load_lora_adapter`，需要
  `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`；配 `load_inplace: true` 会复用同一个 `lora_int_id`，
  `lora/worker_manager.py:263` 那个 `or lora_request.load_inplace` 强制从磁盘重载并替换缓存里的
  adapter——真的是热替换，不用重启 server。
- `entrypoints/openai/chat_completion/protocol.py`：chat-completions 支持 `logprobs` /
  `top_logprobs` / `prompt_logprobs`。

但是**这套东西在这里是多余的，因为 router 的动作根本不是自由生成，是一次 4 选 1 的抽样**。

把 assistant 轮强制以 `{"route": "` 开头，下一个 token 就已经唯一确定了 route。实测 Qwen 分词器
（`_assert_route_tokens` 在构造时对真实 tokenizer 断言，不是假设）：

```
prefix '{"route": "' -> [4913, 8966, 788, 330]
memory  -> 17269  (单 token)
sft     -> 82 's' + 723 'ft'
both    -> 21028  (单 token)
neither -> 811 'ne' + 2485 'ither'
```

`sft` 和 `neither` 虽然是两个 token，但**首 token 两两不同**（17269 / 82 / 21028 / 811），所以在这
一个位置上对这 4 个 logit 做 `log_softmax`，就是完整的策略：一次前向、可微、熵是精确的。这和
`router_policy.sample_action` 里的 `Categorical` 是同一个东西，只是 logits 来自 LLM 而不是
`nn.Linear`。

于是整个采样/重算/同步的问题一起消失：

1. **router 不用 vLLM。** 每个决策 ~0.2s，一个 batch 80 个决策 ≈ 1-2 分钟，而同一个 batch 的
   AppWorld self-eval 是 ~5.8 小时（K=8，2 路并行，70min/次）。推理引擎在这里没有任何可优化的东西。
2. **采样和梯度用同一份权重**，所以重算出来的 logprob 不是"接近"采样时的值，**就是同一个值**。
   `verify_recompute_matches_sample` 直接按 1e-3 断言，实测 `|diff| = 0.00e+00`。跨引擎比较永远做
   不到这么严——它必须容忍的那点误差，恰好能盖住"replay 了错的 prompt""存错了 route index"这类
   真 bug。
3. **`optimizer.step()` 之后没有任何东西需要同步**，没有热加载，没有采样器和学习器不一致的窗口。
4. 省下一张卡（原方案里 router 的 vLLM serving）。

**代价，明说**：router 不能先推理再决定。但**今天这一条代价是 0**——`router_llm_policy.
ROUTER_LLM_SYSTEM` 要求的输出是 `{"route": ..., "rationale": ...}`，`rationale` 在 `route` **之后**
生成，本来就影响不了决策，纯粹是日志。要做 reason-then-decide 得把 vLLM 那条路加回来，而且训练目标
要在"含 rationale 的 logprob"（~100+ token，方差大，而一次 iteration 只有 9 步梯度）和"只算 route
token"（有偏的部分梯度）之间选一个；目前没有任何测量说 CoT 对这个 4 选 1 判断有帮助，所以不做。

system prompt 和 payload 直接从 `router_llm_policy` 原样 import，没有另写一套——这样
`run_router_llm_probe.py` 那个未训练的 route 分布才是这次训练真正的 step-0 基线，而不是一个换了
提示词的表亲。

### 16.2 显存约束下的两段式：采样丢图，更新时重算

一个 batch 有 K=8 × 10 = 80 个决策。**不能**像线性 router 那样把 80 个 logprob 挂在图上求和再一次
backward——那是 80 张 8B 模型的前向图，装不下。所以：

- **采样阶段**（`torch.no_grad()`）：只记 `prompt_ids` + 选中的 `index`（`SampledDecision`），
  probs/entropy 存 detach 到 CPU 的副本做诊断。不留图。
- **更新阶段**：逐个决策重放前向（带梯度），`term = -(adv * logprob) - coef * entropy`，
  **逐个 backward 累加** `.grad`，最后一次 `optimizer.step()`。

逐个 backward 和"求和后一次 backward"在数学上完全等价（和的导数等于导数的和），但同时只需要驻留
一张前向图。loss 的形状和 `train_router_selfreward.run_one_batch_update` 逐字对应，entropy 项也仍然
是在"所有候选 × 所有决策"上求和，所以 v4 标定出来的 `entropy_coef=0.01` 的量级不用重调。

### 16.3 探索够不够：一个差点看错的测量

GRPO 要求 K 个候选真的会做出**不同**的决策。如果策略太确定，8 个候选决策完全一样 → `pass_k` 全同 →
advantage 全是 0 → 梯度是 0，而**日志看起来完全正常**，可以这样空转 40 小时。所以开工前先量。

第一次量出来的结论是错的，记下来因为很容易再犯：拿 `cheap_train_v5` 的真实 record 当输入，测出
Qwen3-1.7B 平均熵只有 0.073 nats、Qwen3-8B 只有 0.037 nats，且 `sft`/`both` 概率≈0，看起来是彻底
坍缩。**但那 20 条 record 里有 0 条同时带 memory 和 sft 两个草稿**（19 条只有 memory）——payload 里
`drafted_sft_plan_candidate: null`，那 `sft`/`both` 本来就没东西可提交，给它们 0 概率是**正确**判断，
不是坍缩。这是典型的"对照没搭对就读出了结论"，和 §13.1 那三条是同一类错误。

补上真实的 Gemini 修复计划（`sft_repair_probe_gemini_v1/plans.json`）让四个 route 都真正可选之后，
同一批任务重测（n=20，T 是给 4 个 route logit 做 softmax 前的温度）：

| 模型 | T | 平均熵 (nats) | p(argmax) | P(8 个候选完全相同) | 平均 route 概率 | argmax 分布 |
|---|---|---|---|---|---|---|
| Qwen3-1.7B | 1.0 | 0.312 | 0.865 | ~0.0000 | mem .064 / sft .131 / both .049 / neither .755 | neither 15, sft 3, memory 2 |
| Qwen3-8B | 1.0 | 0.154 | 0.931 | 0.0033 | mem .177 / sft .000 / both .136 / neither .686 | neither 13, memory 4, both 3 |
| Qwen3-8B | 3.0 | 0.417 | 0.832 | ~0.0000 | mem .190 / sft .003 / both .131 / neither .676 | 同上 |

两个模型在 T=1.0 都有足够探索（8B 是 99.7% 的 batch 会出现分歧），不需要调温度。

**反直觉的一条，值得单独记**：模型越大越确定、探索越少（8B 的熵只有 1.7B 的一半）。"大模型基础判断
更好"这个直觉在这里要和"大模型更不肯探索"对冲，不是单向的。

**选 Qwen3-8B**，理由是基础判断质量本来就是换掉 1.7B 的唯一动机，而探索量（P=0.003）够用。
**已知缺口**：8B 在 T=1.0 下几乎不单独选 `sft`（p≈0.000），虽然通过 `both`（p=0.136）仍然会提交 sft
产物。纯 `sft` 这个动作实际上是半死的，靠熵正则和训练能不能救回来，留作观察项。
`policy_temperature` 作为配置项保留（默认 1.0），是运行中发现坍缩时唯一不动权重就能加宽探索的杠杆。

### 16.4 学习率：实测出来的问题和预期相反

交接文档担心的是"lr=0.01 是给 144 参数线性模型调的，对 LoRA 肯定不对"，并提示 LoRA 常用 1e-4~1e-5。
实测（`smoke_router_llm_policy_gradient.py` 的扫描，单个决策、advantage=+1.0 反复推）方向是反的：
**1e-4 太大，不是太小。**

| lr | 一步之后 Δp(route) | 9 步之后 p(route) |
|---|---|---|
| 1e-6 | -1.0e-13 | 0.00000 |
| 1e-5 | +5.4e-10 | 0.86551 |
| 5e-5 | +5.8e-06 | 1.00000 |
| 1e-4 | **+9.7e-01** | 1.00000 |
| 5e-4 | +1.0e+00 | 1.00000 |

lr=1e-4（就是本仓库 `train_qwen_lora.sh` 的 SFT 学习率）**一步就把某个 route 的概率从 0.00000 推到
0.949**；同样的 lr 下熵正则也发散——10 步之后熵从 0.113 掉到 0.0001（完全坍缩），而不是升向均匀。
注意这不是符号错：lr 从 1e-7 到 3e-5 熵都是**上升**的（0.113 → 0.149 / 0.425 / 0.686 / 0.792），
只有 1e-4 炸掉。所以那次 smoke 的 FAIL 是步长问题，不是梯度方向问题。

动力学高度非线性，原因是 LoRA 的 B 初始化为 0：前几步几乎不动（`delta` 在 1e-10 量级），B 一旦起来
就非常快。所以"一步移动多少"和"九步移动多少"完全不是线性关系，只看单步会严重低估。

**定 lr=1e-5**，并且默认开 `--max-grad-norm 1.0`。开 clipping 的理由不是常规保险，而是这里
**一次 iteration 只有 9 次更新，没有从一次坏更新里恢复的机会**，而 §12 已经确认坍缩掉的 router 只能
重新初始化、不能靠继续训练爬回来。真实 batch 的 advantage 是 80 个混合符号的决策，净推力远小于上表
那种"同一个决策连推 9 次"的最坏情况，所以 1e-5 可能还偏小——`train_log.jsonl` 每个 batch 记
`mean_prob_shift` / `max_prob_shift` / `grad_norm`，第一个真实 batch 之后按这几个数再定。

### 16.5 冒烟测试（都在真实训练启动之前跑过并通过）

- `scripts/smoke/smoke_router_llm_logprob.py`：分词器四路可区分、prompt 与 `decide_route` 逐字节
  一致、分布良构（概率和为 1、熵在 [0, ln4]）、**重算 logprob == 采样 logprob**（实测 |diff|=0）、
  梯度只进 LoRA 不进 base（0 个 base 张量带梯度）。
- `scripts/smoke/smoke_router_llm_policy_gradient.py`：正 advantage 抬高该 route 概率、负 advantage
  压低、只有 LoRA 权重变、熵正则把熵从 0.113 推到 1.017 nats、以及 §16.4 的 lr 扫描。
- `scripts/smoke/smoke_router_llm_batch_e2e.py`：只 stub 三个边界（writer LLM、AppWorld 评测子进程、
  sft guided replay），跑**真实的** `run_one_batch_update`——每个决策都带 `SampledDecision`（确认真的
  走了 `trained_llm` 模式，没有静默退化成零张量）、advantage 组内和为 0（实测 1.1e-16）、候选确实分
  歧、LoRA 真的动、诊断字段齐全。

**这里踩到一个 stub 设计的坑**：第一版 stub 的记忆内容是 `"stub memory fact number 3 about topic 3"`
这种占位文本，router 以概率 1.0 判 `neither`——这是**对垃圾内容的正确判断**，但它让熵、logprob、梯度
全变成 0，那条"候选是否分歧"的断言就变得没有意义了。stub 内容必须和 `routed_writer_system` 真实产出
同构（具体的 API 事实 + 具体的修复步骤）才测得到东西。另外那条断言本身也和规模有关：在测试用的
K=4 × batch_size=4、p(argmax)=0.99 下，16 个决策全同的概率本来就有 ~85%，所以断言分歧等于在测
batch size；真正的探索量是 §16.3 在真实规模上单独量的，e2e 这边把温度设成 3.0，测的是"有熵的时候
分歧能不能传到 advantage 和梯度"这条通路。
