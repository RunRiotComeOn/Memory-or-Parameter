# ALFWorld 持续学习消融汇总

*截至 2026-09-22。数据来源见每节末尾的产物路径；方法与逐步过程记录在 `running_log.md` §15。*

## 消融配置方式（可复用到其他 task stream）

> 在同一份固定的无记忆基础轨迹池上，把**产物通道**（记忆 bank / SFT 权重 / 两者都写 / 都不写）与
> **筛选策略**（router 判断 vs 强制全写）交叉成网格，每个格子产出的产物冻结成一个静态状态，再用
> 同一个 serving 副本、同一批题分别重放「训练集内」和「分布外」两条线，全部对同一个无记忆底座
> 基线作差，并对每个配置单独量化自我泄露。

展开成六条可执行的约束：

1. **基础轨迹池只跑一次**，所有配置共用（这里是 `base_train_v2`，200 题 train split，无记忆）。
   配置之间唯一的差别是从这同一批轨迹里写出了什么产物，不是轨迹本身。
2. **产物通道和筛选策略正交**。通道决定产物落到哪里（bank 走检索、SFT 走权重），策略决定写多少
   （router 四选一 vs 无条件全写）。两者不能混在一个变量里，否则分不清收益来自筛选还是来自通道。
3. **冻结后再评测**。建 bank / 训 LoRA 全部结束、bank 不再变、权重 merge 完成之后，才开始重放；
   评测过程中不再有任何写入。
4. **两条评测线都要跑**：训练集内（就是那 200 题本身）测「在训练分布上学到了没有」，分布外
   （`valid_unseen` 57 题，房间布局在 train 里完全没出现过）测「能不能泛化」。两条线可能给出
   相反的结论——本轮的叠加效应就是如此。
5. **基线必须同副本、同配置、同 seed**。跨副本/跨配置的绝对分数不可比（`running_log.md` §12.2(b)
   在 AppWorld 上因此推翻过两次结论）。本轮实测 ALFWorld 的 rollout 是逐题确定性的（见下），
   但这个前提每换一个 benchmark 都要重新验证一次，不能假设。
6. **每个配置单独量化自我泄露**。bank 和 SFT 数据都是从被测任务自己的轨迹里写出来的，训练集内
   那条线天然有循环论证风险，而且不同通道的泄露量差一个数量级（见下表「训练时见过自己轨迹」列）。

---

## 主结果：分布外（valid_unseen 57 题）

这 57 题从未参与建 bank 或 SFT 数据的构建，房间布局在 train split 里也没出现过，**无泄露**。
这是本轮的主指标。

| 配置 | memory bank | SFT 模型 | pass_rate | vs 基线（逐题） |
|---|---|---|---|---|
| 无记忆基线 | — | 底座 | 0.7544 (43/57) | — |
| 强制全写 | 143 条 | 底座 | 0.8246 (47/57) | 10 升 6 降 (+4) |
| **router 仅记忆** | **34 条** | 底座 | **0.8947 (51/57)** | 9 升 1 降 (+8) |
| router 仅 SFT | — | 21 条样本 | 0.8596 (49/57) | 8 升 2 降 (+6) |
| router 记忆 + SFT | 34 条 | 21 条样本 | 0.8421 (48/57) | — |
| **强制全 sft** | — | **181 条样本** | **0.9825 (56/57)** | 13 升 0 降 (+13) |

关键两两对比：

```
router sft(21)   -> force sft(181) :  7 升  0 降   net  +7
router mem(34)   -> router mem+sft :  2 升  5 降   net  -3
router sft(21)   -> router mem+sft :  3 升  4 降   net  -1
```

## 训练集内（train 200 题）

bank 和 SFT 数据都出自这 200 题自己的轨迹，**存在自我泄露**，按配置分别量化后读。

| 配置 | 整体 | 训练时见过自己轨迹 | 去泄露后（本配置 / 同子集基线） |
|---|---|---|---|
| 无记忆基线 | 0.690 (138/200) | 0 | — |
| 强制全写 143 条 | 0.785 (157/200) | 0 | 0.785 / 0.690 |
| router 34 条记忆 | 0.810 (162/200) | 0 | 0.810 / 0.690 |
| router 仅 SFT (21) | 0.840 (168/200) | 21 | 0.8212 / 0.7151（179 题） |
| **router 记忆+SFT** | **0.880 (176/200)** | 21 | **0.8659 / 0.7151**（179 题） |
| 强制全 sft (181) | 0.945 (189/200) | **181** | 0.5789 / 0.0526（仅剩 19 题）※ |

※ 去掉 181 题泄露后只剩 19 题，且这 19 题恰好是 guided replay 都没跑成功的最难子集（基线 1/19），
样本量小且有选择偏差，**这个格子的训练集内数字不可用**，它的可信证据是分布外的 0.9825。
记忆侧的泄露此前已单独核查过：v4 的 32 个翻正里只有 1 个检索到了自己贡献的条目（`running_log.md` §15）。

关键两两对比：

```
baseline         -> router sft(21)  : 37 升  7 降   net +30
baseline         -> router mem(34)  : 32 升  8 降   net +24
router mem(34)   -> router mem+sft  : 18 升  4 降   net +14
router sft(21)   -> router mem+sft  : 10 升  2 降   net  +8
router sft(21)   -> force sft(181)  : 21 升  0 降   net +21
```

---

## 三条结论

### 1. 筛选的价值取决于产物进哪条通道——两侧方向相反

```
记忆侧： router 34 条 (0.8947)  >  强制全写 143 条 (0.8246)     筛选赢
SFT 侧： router 21 条 (0.8596)  <  强制全训 181 条 (0.9825)     数量赢
```

机制上自洽：`--memory-top-k 3` 让检索位成为**稀缺资源**，未筛选的 143 条会把低质量条目挤进那 3 个
位置、顶掉本该命中的好条目——表现为 force_memory 的 10 升 **6 降**。而 SFT 改的是权重，**不占任何
检索位**，多训的样本不排挤已学到的东西——表现为 `router sft → force sft` 在两条线上都是 **0 降**
（分布外 7 升 0 降、训练集内 21 升 0 降）。

**对后续 task stream 的含义**：router 的筛选只在「产物要竞争一个有限的注入预算」时才有价值。
凡是写进权重、不参与检索竞争的产物，应当默认全量收集而不是让 router 挑。

> **⚠️ 2026-09-23 补充的边界条件（见 `scienceworld_summary.md` 结论 1、2）**
> 上面这条规则是只有 ALFWorld 一个 benchmark 时写的，ScienceWorld 跑完后发现它缺了一个前提。
> 「筛选赢」需要**两个条件同时成立**：产物要竞争检索位（通道条件），**并且**基础轨迹池能产出
> 值得筛的素材（质量条件）。ALFWorld 基线 0.69 时两者都满足；ScienceWorld 基线 0.16 时第二个
> 不满足，router 如实判断 168 个失败轨迹「无可提炼」，200 题只写了 3 次 memory，筛出 2 条活跃
> 记忆——分布内外都只是噪声，而强制全写的 159 条拿到 +8.8pp：
> ```
> ALFWorld      基线 0.69：router 34 条 (0.8947)  >  强制全写 143 条 (0.8246)   筛选赢
> ScienceWorld  基线 0.16：router  2 条 (0.1754)  <<  强制全写 159 条 (0.2456)   筛选输到只剩噪声
> ```
> 筛选没有筛出精华，是筛到了什么都不剩。**操作上：先看基础池的 pass_rate，偏低时记忆侧直接用
> 强制全写建立厚度，router 筛选留到基础能力足以产出可提炼素材之后再启用。**
>
> 同理，本节「SFT 侧数量赢」也有前提：真正的决定项是样本里**巩固 vs 修复**的构成，不是数量。
> ALFWorld 的 181 条里 138 条是巩固样本，ScienceWorld 的 48 条里只有 29 条巩固、19 条修复，
> 结果样本更多反而泛化更差（48 条 0.2281 < 13 条 0.2456）。

### 2. 记忆和 SFT 叠加不是相加，且两条线结论相反

- 分布外：叠加 0.8421，**低于**两者单独用（0.8947 / 0.8596），相对各自净 −3 / −1
- 训练集内：叠加 0.880（去泄露 0.8659），**高于**两者单独用，相对各自净 +14 / +8

一个未经验证的猜测：bank 里的记忆是为**底座模型**写的，在训练分布内的任务上内容高度对口，即使模型
行为已被 SFT 改变仍然有用；到了分布外任务上本就只是弱相关，再遇上行为已变的模型就更容易变成干扰。
**要验证需要拿 SFT 之后的模型重新写一遍 bank 再测**，本轮没做。

### 3. ALFWorld 的 rollout 是逐题确定性的，跨天跨副本可比

2026-09-22 在 8030（底座模型）上重测无记忆 valid_unseen 基线：0.7544（43/57），与 2026-09-19 在另一个
副本上测的 `baseline_valid_unseen_v1` **57/57 逐题完全一致**（success、termination_reason、步数全同）。

所以 `running_log.md` §12.2(b) 那条「跨副本绝对分数不可比、差距可达 10-20pp」的警告**只适用于
AppWorld**，本文件所有数字可以直接横向比较。产物：`alfworld_experiment/baseline_unseen57_recheck_0921/`。

---

## 各配置的产物与参数

### 共用基础

- 轨迹池：`alfworld_experiment/base_train_v2/`（200 题 train，无记忆，pass_rate 0.69）
- 分布外评测集：`alfworld_experiment/baseline_valid_unseen_v1/` 的同一批 57 题 task_ids
- 评测参数（所有配置一致）：`max_steps=40`、`seed=20260822`、`--memory-top-k 3`、`max-parallel 4`、
  temperature 0、thinking disabled
- task agent：Qwen3.5-35B-A3B，确定性 serving（`--max-num-seqs 1 --enforce-eager --no-enable-prefix-caching`）

### router 筛选（v4）

- 建 bank：`alfworld_experiment/router_llm_probe_v4/`
  - `route_counts = {'memory': 37, 'sft': 23, 'both': 2, 'neither': 138}`，活跃记忆 **34 条**
  - sft：25 个候选 replay，**21 条验证通过**（yield 0.84）
- 仅记忆重放：`alfworld_experiment/freeze_replay_v4_200/`、`router_llm_probe_v4/eval_valid_unseen57/`
- SFT LoRA：`alfworld_experiment/router_v4_lora/`，merged 至 `/nas04/yixuh/router_v4_merged`
- 仅 SFT / 记忆+SFT 评测：`alfworld_experiment/routerv4_{sftonly,both}_{unseen57,train200}/`

### 强制全写记忆（force_memory）

- `alfworld_experiment/router_force_memory_v1/`：199/200 走 memory，**143 条**活跃（存活率 71.9%）
- 重放：`alfworld_experiment/freeze_replay_forcemem_200/`、`router_force_memory_v1/eval_valid_unseen57/`

### 强制全 sft（force_sft）

- `alfworld_experiment/router_force_sft_v1/`：199/200 走 sft，**活跃记忆 0 条**（bank 全程为空）
  - 199 个候选 replay，**181 条验证通过**（yield 0.9095）
  - 拆分：来自原本成功任务的「巩固」138 条，来自原本失败任务的「修复」61 条
- SFT LoRA：`alfworld_experiment/force_sft_lora/`，merged 至 `/nas04/yixuh/force_sft_merged`
- 评测：`alfworld_experiment/freeze_replay_forcesft_200/`、`alfworld_experiment/forcesft_eval_valid_unseen57/`

### SFT LoRA 训练

`scripts/train_agent_sft_lora_peft.py`（transformers + peft，**不是** `router_sft_lora_update.sh`
的 swift + deepspeed 路径，后者在这台机器上连续失败 6 次，见 `running_log.md` §15）。

- 超参沿用原脚本：r=8、alpha=32、`all-linear`、lr=1e-4、3 epoch、bs=1、gradient checkpointing
- 可训练参数 11,238,720（MoE 的 256 个专家是融合 3D `nn.Parameter` 而非 `nn.Linear`，`all-linear`
  够不到它们，LoRA 落在 linear_attn / self_attn 投影、shared_expert 和 MoE 门控上）
- `max_length=6144`（原脚本的 3072 是被 OOM 逼出来的妥协）。**⚠️ 这个值当时按「最长样本 5,410
  tokens」定的，那是只看 `router_llm_probe_v4` 的 21 条得出的；`force_sft` 的 181 条里最长有
  8,462 tokens，所以少数样本被截断了**。更糟的是当时的截断保留尾部、丢弃前缀，会把系统提示整个
  切掉，那些样本不再以 system 轮开头。已在 2026-09-23 修成「保留系统轮 + 尾部」并把默认上限提到
  12,288（见 `scienceworld_summary.md`）。**本文件的 `force_sft(181)` 结果是在修复前跑的，带有
  这一已知瑕疵，未重跑**；影响面应该很小（p90 仅 3,885），但没有量化过
- 显存峰值 **36–39.5GiB/卡**（GPU 6,7，`device_map="auto"`）
- loss 走势：force_sft(181 条) 0.0454 → 0.0258 → 0.0143；router v4(21 条) 0.0443 → 0.0138 → 0.0067
  （起点本就接近 0——SFT 目标是模型自己 replay 成功的输出，是 rejection sampling 的固有性质）

### merge

`scripts/rekey_lora_to_full_model.py` → `scripts/merge_peft_lora.py`（CPU）。

LoRA 在 `AutoModelForCausalLM`（`model.layers.*`）上训练，而 vLLM 服务的是
`Qwen3_5MoeForConditionalGeneration`（`model.language_model.layers.*`），**直接 merge 会匹配不到任何
键且不报错**，静默产出一个与底座完全相同的模型。re-key 脚本插入 `language_model.` 并把
`target_modules` 从 13 个后缀改写成 350 条显式全路径（后缀匹配会误伤 vision 塔和 MTP 里的同名模块）。

merge 后逐张量校验：目标模块 rel 3.4e-3（router v4）/ 7.3e-3（force_sft），vision 塔与融合专家权重
逐位未变（rel = 0）。
