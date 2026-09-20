# Router 训练跑动记录

简明记录每次跑动的 reward 定义、关键参数、结果。详细设计动机见 `router_reward_v1/DESIGN.md`。

---

## 1. `train_router_grpo.py`（原始设计，DESIGN.md §9）

- **reward**：每个 checkpoint 在完整 57 题真实 dev 集上跑一次，reward = 这次 pass_rate − 上一个 checkpoint 的 pass_rate（组内均值做 baseline，不用历史最高）。
- **参数**：G=4，每条链切 B=3 个 batch（每 30 题一个 checkpoint），lr=0.05，无熵正则。
- **产物目录**：`router_reward_v1/checkpoints/router_iter{1,2,3}.pt`
- **结果**：
  - iter1：四条链 [0.368, 0.456, 0.526, 0.474]，均值 0.456
  - iter2：均值 0.408（比 iter1 掉）
  - iter3：均值 0.456（回到 iter1 水平）
  - 三轮下来没有单调趋势，方差盖过了信号，判定不出训练有没有在起作用。

## 2. 便宜 reward 可行性 pilot（`pilot_inbatch_v1`）

- **目的**：不训练，只验证"batch 自测 + 探针集"这套便宜信号靠不靠谱。
- **发现 1（dedup 修复前）**：用 `router_iter3.pt` 采样 4 个候选，self/probe reward 几乎全负；查出 router 对 base_agent 本来就成功的题也照样写记忆（8/10 已成功的题一个 neither 都没选），且 v1 的 `memory_operation` 只有 `add`、没有 `refine`，导致同场景变体反复写出近乎重复的记忆、挤占 BM25 top-3 检索位。
- **修复**：`router_bank_builder._dedup_against_active_bank`——内容生成后自动按 `topic_overlap` 相似度把 `add` 转成 `refine`，不需要 router 学习"选哪条旧记忆"。
- **发现 2（dedup 修复后，`router_reward_v1/pilot_inbatch_v1`）**：4 个候选，self reward 和 probe reward 相关系数 0.905（n=4，样本很小但方向一致）——说明这套便宜信号不是纯粹的自我泄漏假象，值得拿来训练。

## 3. `train_router_cheap_reward.py`（v2，probe reward，per-batch 更新）

- **reward**：固定 15 题 dev 探针集（`noise_serial_v1/run_a` 里选的前 15 题），probe reward = 当前 batch 后的 probe pass_rate − 上一个 batch 的 probe pass_rate。**每个 batch 后都做一次梯度更新**（不是攒完整个 iteration 才更新）。
- **参数**：K=4 rollouts/batch，B=9 个 batch（每 10 题），lr=0.05，无熵正则。
- **产物目录**：`router_reward_v1/cheap_train_v2/`
- **结果（iteration 1）**：
  - 最终验证：`full_dev_pass_rate = 0.3684`（21/57），**低于空 bank 基线 0.4386**。
  - 拆分：probe_subset(15)=0.333，held_out(42)=0.381——探针集反而更差，不是过拟合探针集的问题。
  - **route 分布完全坍塌成确定性规则，且方向反了**：`base_agent 成功 → 100% both`（硬塞记忆），`base_agent 失败 → 100% sft`（sft 在 v1 里不写 bank、也不做真实微调，等于什么都没干）。

## 4. `train_router_selfreward.py`（self reward，per-batch 更新，K=4）

- **改动**：放弃固定探针集，reward 改成"用当前 batch 自己的题 replay"（即诊断用的 self_reward 提升为正式训练信号），因为固定探针集有"永远盯着同一批 15 题"的过拟合风险。self+probe 不再需要同一候选双评测，改成两个候选的 self 评测分别占两个 server 并行（一个 batch 4 轮变 2 轮）。
- **参数**：K=4，B=9，lr=0.05，无熵正则。
- **产物目录**：`router_reward_v1/cheap_train_v3_g4/`（已归档，原名 `cheap_train_v3`）
- **结果（iteration 1）**：
  - 最终验证：`full_dev_pass_rate = 0.4035`（23/57），仍低于基线 0.4386，但比 v2 好。
  - 拆分：probe_subset(15)=0.4667，held_out(42)=0.381。
  - **同样完全坍塌**，但坍塌方向不同：90 道题**全部**选 `both`，不管 base_agent 成功还是失败，一次 neither 都没有。

## 5. `train_router_selfreward.py`，K=8（本机在跑）

- **改动**：只改了 `--rollouts-per-batch` 从 4 提到 8，想验证"是不是 G 太小导致的坍塌"。reward/参数其余不变（lr=0.05，无熵正则）。
- **参数**：K=8（4 轮×2-way 并行/batch），B=9，lr=0.05。
- **产物目录**：`router_reward_v1/cheap_train_v3/`
- **状态**：进行中（本机，两个确定性 vLLM server）。跑到 batch 5/9 时看 route 分布还没有像 G=4 那次那样一路收敛到单一 route，好坏参半，尚无定论。**预判**：G=8 只降低单步估计的方差，不改变"9 步、无正则、lr=0.05"这个坍塌的根本机制，怀疑大概率还是会坍塌，只是可能没那么快/那么彻底。

## 7. `train_router_selfreward.py`，K=8（本机，已完成）

- **改动**：同 §5，只改了 `--rollouts-per-batch` 从 4 提到 8，reward/lr/无熵正则均未变。
- **参数**：K=8（4 轮×2-way 并行/batch），B=9，lr=0.05。
- **产物目录**：`router_reward_v1/cheap_train_v3/`
- **结果（iteration 1，已完成）**：
  - **`full_dev_pass_rate = 0.5789`（33/57）——目前唯一一次超过空 bank 基线（0.4386）的结果。**
  - 拆分：probe_subset(15)=0.5333，held_out(42)=0.5952。
  - route 分布：`{'memory': 82, 'both': 8}`，active_entries=30。仍然坍塌成近乎确定性策略（`memory` 一家独大），但方向没有反：这次没有陷入"成功题也硬塞记忆"或"失败题什么都不写"，而是几乎全部走纯 `memory`，很少用 `both`，一次 `neither`/`sft` 都没有。
  - **结论**：G 越大越接近这次的方向，但仍然坍塌（只是没坍塌到错误方向），说明 K=8 只是缓解了单步方差，没有解决根本的训练动力学问题（见 §6 v4 的熵正则）。这次意外获得正结果，不能确定是"router 学到了东西"还是"坍塌方向本身恰好对"——待 §8/§9 的改动（reward 改成含无记忆基线的组内均值 + router 能看到实际内容）上线后再对比。
  - **⚠️ 2026-09-18 追加，重要更正**：DESIGN.md §13（另一台机器的三方后端对比）证实**跨副本/跨运行的绝对分数不可比，差距可达 10-20pp**。核对发现这里用来说"超过基线"的 0.4386 来自 `noise_serial_v1/run_a`，生成于 **2026-08-29**——比 §11 搭建两副本确定性 serving（2026-09-12）还早，几乎肯定不是同一套服务端配置；而 0.5789 是在这次两副本里的 `127.0.0.1:8000` 上测的。**这俩数字不能直接比**，"G=8 超过基线"这个结论目前**不成立，需要重新验证**。已经在同一个副本（8000 端口，同 seed=20260822，同 57 题 dev）上补跑一次匹配的无记忆基线（`router_reward_v1/baseline_recheck/`）。
  - **✅ 2026-09-18 同条件复核结果**：同副本（127.0.0.1:8000）无记忆基线 `pass_rate = 0.4912`（28/57）。G=8 的 0.5789（33/57）与它同副本、同 seed、同 57 题——**这次是真正可比的对照，G=8 确实比无记忆基线高 8.77 个百分点**，"router 学到的路由/记忆内容有正效果"这个结论目前站得住（n=1 次 iteration，仍需更多次跑动验证是否稳定）。

## 6. `train_router_selfreward.py` + 熵正则（已取消）

- **改动**：在 loss 里加熵正则项 `loss = pg_term − entropy_coef × entropy_sum`（`entropy_coef` 默认 0.01），直接鼓励 route 分布不要过早收敛成确定性策略；同时把 `lr` 从 0.05 降到 0.01。验证阶段（贪心）也顺手记录 `mean_entropy`，即使 argmax 还没变、也能提前看到底层分布在不在塌。
- **动机**：v2、v3(G=4) 两次独立训练（不同 reward 定义）都坍塌成确定性策略，且实测 v3(G=4) 的采样熵从 1.14 nats 掉到 0.65 nats（4 分类最大熵 ln4≈1.386）——判定是训练动力学问题（无正则、更新次数少、线性模型容易饱和），不是换 reward 定义能单独解决的。
- **产物目录**：`router_reward_v1/cheap_train_v4/`（另一台机器写，共享此 NFS 目录）
- **状态**：已取消。

## 8. 2026-09-18 代码改动（v5，详细动机见 DESIGN.md §14）

三处相关改动：

1. **reward**：先改成含无记忆基线的 K+1 路组内均值（修复此前 `self_baseline` 在代数上被组内均值抵消、从未真正进入梯度的 bug），**后又撤回**——代数展开发现基线在 K+1 均值里只占 `1/(K+1)` 权重，K=8 时只有 1/9，信号被稀释得太弱，方向对但强度可能不够。当前用回最原始的纯组内相对 advantage（`advantage_k = pass_k - mean_j(pass_j)`），`self_baseline`/`rewards` 只做日志展示，不进梯度。还没想到更好的加权方式，先不加。
2. **router 特征**：`FEATURE_DIM` 从 5 变 35。删掉 `frac_position`/`frac_remaining`（对固定 90 题的线性模型是精确冗余）。新增两块哈希词袋特征——最近两个 batch bank 新增/变动条目的实际文本、这道题已起草候选内容的实际文本。配套把 `router_bank_builder.run_router_chain` 改成"先起草内容、router 再看草稿决定路由"，而不是先路由再让 LLM 照写。代价：每道题都要起草一次内容，包括最后路由是 `neither` 的题。新增 `ROUTER_DISABLE_CONTENT_FEATURES=1` 开关，可以让这两块特征归零（对线性模型等价于特征不存在），用于对照实验，不用复制代码。
3. **SFT 真正训练进去**：以前 `sft_plan` 只记一句 `repair_target`，从未被消费——route=sft 和 route=neither 在 reward 上完全等价。现在换专门的 `appworld_sft_writer`（同一个基座模型换 prompt，不是教师模型）产出自然语言修复计划，只在失败题上调用；委托 `scripts/run_appworld_guided_replay.py` 把计划当 `memory_block` 注入、在真实环境里重新跑一次这道题，只有 AppWorld 自己判定成功的 replay 才把**真实对话记录**（已把 guidance 文本从训练样本里切掉，避免训练/推理提示不一致）存进训练池；池子跨过 8 的倍数就触发 `scripts/router_sft_lora_update.sh`（LoRA 训练 + merge + 两个副本都重载新 checkpoint，保持副本对等；训练子进程用独立进程组跑，超时/失败会整组 kill 并自动把两个副本拉回未微调的 base 模型，不会留下服务器全挂的状态）。
4. 预算估算加上了 sft guided-replay 和 LoRA 重训的粗略耗时（标了 UNVERIFIED，这条链路还没真正跑过）。

**已知未修的问题（不是这次要解决的）**：`self_baseline` 一旦某次 SFT 训练真的触发，后面所有 batch 用的都是微调后的 agent，但 baseline 还是原始 base agent 的分数，比较会悄悄失真，目前没有自动重新测基线的逻辑。

## 9. `train_router_selfreward.py` v5（本机在跑，2026-09-18 06:29 UTC 启动）

- **改动**：§8 的三处改动全部生效后的第一次正式训练。
- **参数**：K=8，B=9，lr=0.01，entropy_coef=0.01，`--base-url` 8000/8001（本机两个确定性副本）。
- **产物目录**：`router_reward_v1/cheap_train_v5/`，日志 `router_reward_v1/train_v5.log`，tmux session `train_router_v5`。
- **状态**：进行中，预计约 49.6h/iteration（42h self-eval + ~2.1h sft guided-replay + ~2h LoRA 重训，后两项是没跑过的估计）。

## 10. `train_router_selfreward.py` v5_nofeat 消融（另一台机器，与 §9 同时起跑）

- **改动**：唯一变量 `ROUTER_DISABLE_CONTENT_FEATURES=1`——router 看到的两块哈希内容特征强制归零，内容照样起草/写入，只是不给 router 当特征看。其余参数与 §9 完全一致（K=8、lr=0.01、entropy_coef=0.01、reward 公式、SFT 训练开关）。
- **目的**：判断今天新加的"router 能看到实际内容"这个改动（§8.2）到底有没有用——这是今天成本最高（每题多一次起草调用）但从没验证过的改动。
- **产物目录**：`router_reward_v1/cheap_train_v5_nofeat/`（`ROUTER_OUTPUT_DIR` 环境变量指定，跟 §9 共享同一份 NFS 代码但输出目录不冲突），日志 `router_reward_v1/train_v5_nofeat.log`，tmux session `train_router_v5_nofeat`。
- **本机（第二台）实际配置**（2026-09-18 00:03 PDT 启动）：这台机器上 8000/8001/8002 已被别人的服务占用（8000 是另一个用户的 qwen3-vl vLLM，8001/8002 是别人的 judge_server），GPU 0/2/3 也有别人的进程。所以两个确定性副本改成 **8010（GPU 4,5）/ 8011（GPU 6,7）**，tmux session 名 `det_server_nofeat_a` / `det_server_nofeat_b`（不用默认的 `det_server_a/b`，否则会跟 §9 那台机器往同一份 NFS 日志 `appworld_experiment/det_server_a.log` 里混写）。训练进程里相应导出 `SERVER_A_NAME/SERVER_B_NAME/SERVER_A_PORT/SERVER_B_PORT/SERVER_A_GPUS/SERVER_B_GPUS`，这样 §8.3 的 SFT LoRA 重训 + 双副本重载（以及失败时的 `_restore_base_servers` 兜底）打的是本机真实的端口/GPU，不是 8000/8001 和 GPU 0,1/2,3。
- **⚠️ APPWORLD_ROOT 必须隔离**：`train_router_selfreward.py` 生成的 AppWorld experiment 名字（`cheap_v4_iter{i}_b{b}_k{k}_self`）在 v5 和 v5_nofeat 两个 arm 之间**完全相同**，而 AppWorld 把 per-task DB / evaluation / logs 写在 `$APPWORLD_ROOT/experiments/outputs/<experiment_name>/tasks/<task_id>/`，两个 arm 的 task_id 也一样。两台机器共享 `/nas04`，如果都用默认的 `APPWORLD_ROOT=/nas04/yixuh/appworld_root`，就会并发写同一批 task 目录、互相踩 DB 和评测结果。所以本机改用 `APPWORLD_ROOT=/nas04/yixuh/appworld_root_nofeat`（`data/` 从原 root 整份拷了一份 194M，`experiments/outputs` 和 `.tmp` 全新空目录）。后续任何"同一脚本在两台机器上并行跑不同 arm"的实验都要记得这一条。
- **状态**：进行中（预算与 §9 相同，~49.6h/iteration）。`git pull` 在本机失败（`No user exists for uid 1644066`，ssh 取不到 passwd entry），但工作区就是 §9 那台机器在写的同一份 NFS checkout，已经在最新 commit `cb7d00b` 上，不影响跑动。

## 11. `probe_sft_repair_yield_gemini_teacher.py`（Gemini teacher 写修复计划，本机，已完成）

- **改动**：`scripts/probe_sft_repair_yield.py` 的姊妹脚本，唯一变量是修复计划由谁写——原版是 qwen35-tau 自己给自己写（`appworld_sft_writer`），这版换成 **Gemini（`gemini-3.1-pro-preview`）当 teacher**，执行修复的 student agent 仍然是本机 qwen35-tau，只是替它起草计划的模型换了。计划文本仍然是自然语言、注入 `memory_block` 走 guided replay，不是脚本回放。
  - 踩坑记录：`--teacher-model` 一开始猜的 `gemini-3-pro-preview` 返回 404（已停用），API 报错里直接给出替代型号 `gemini-3.1-pro-preview`，改过来后正常。
- **题目**：跟原版完全一样的 33 道 `base_train_v2` 失败题（18 个不同模板），保证跟以后任何自写手版本的结果可比。
- **产物目录**：`router_reward_v1/sft_repair_probe_gemini_v1/`，日志 `router_reward_v1/sft_repair_probe_gemini.log`。
- **结果（33/33 全部跑完）**：
  ```
  self-healed（不用任何计划，control 重跑自己就过了）：9/33
  真正还在失败的分母：24
  Gemini teacher 救回：15/24 = 62.5%
  被计划弄坏的（control 过了、repair 反而失败）：1

  按难度：1(易) 4/6=67%　2(中) 6/7=86%　3(难) 5/11=45%
  按原始终止方式：max_steps 2/4　repeated_truncation 0/1　task_completed 13/19=68%
  ```
  中途 18/33 时的中间值是 76.9%，后面进来的多是 difficulty=3 的难题，把最终数字拉到了 62.5%——难度越高救回率越低，方向符合预期。
- **⚠️→✅ 缺的对照已经补上，见 §12**：写这节时自写手版本（`sft_repair_probe_v1/`）只跑完 1 道 repair，所以当时只能说"Gemini 效果不差"。现在自写手版已跑完，同 15 道题的配对比较是 **Gemini 8/15 vs 自写手 1/15**，"比自写手更好"这一点坐实了。

## 12. `probe_sft_repair_yield.py`（自写手版修复产出率，本机，已完成）

- **目的**：把整条 SFT route 赖以成立的那个数单独量出来——**给一道失败题一份修复计划，重跑能救回百分之几**。此前它埋在训练循环里，只对"router 恰好把 sft/both 路由到的失败题"可见，全部证据只有 v5 的 `2 replayed, 0 verified`，而那两条还是同一个 task 模板（`22cc237`）的变体，是轶事不是样本。
- **改动**：跟 §11 的 Gemini 版**唯一变量**是写计划的人——这版由 qwen35-tau 自己写（`appworld_sft_writer.APPWORLD_SFT_WRITER_SYSTEM`），也就是让**失败的那个模型自己诊断自己**。执行修复的 student agent、33 道题、control arm、summary 结构全部相同。
- **两个 arm，都在今天同一批服务端上跑**：
  - `control`：原样重跑，**无记忆、无 guidance** —— 今天真实可比的分母。
  - `repair`：同一道题，注入 writer 的修复计划。
  - **control 不是可有可无的**：那 33 条失败记录来自 `base_train_v2`（2026-08-28），远早于 §11 两副本确定性 serving（09-12）。DESIGN.md §13 已确认跨配置绝对分数不可比（10-20pp），§7 那条更正就是因为拿旧配置的数比新配置栽过一次。拿今天的 guided replay 去比五周前的失败记录，是同一个错误再犯。
- **参数**：k=1、temperature=0、固定 seed 20260822、`max_tokens=4096`（writer 与 agent 均为当天统一提升后的值）、副本 8010/8011（repair arm 双路并行，按索引奇偶分流；`--max-num-seqs 1` 保证并发请求排队而非拼 batch，确定性不受影响）。
- **产物目录**：`router_reward_v1/sft_repair_probe_v1/`，日志 `router_reward_v1/sft_repair_probe_v1.log`。
- **结果**：
  ```
  33 道 → writer 挂 5 道（见下）→ 评了 28 道
  self-healed（今天无计划就通过）  11/33
  真实分母（今天仍失败且已评）     17
    RESCUED                        1  = 5.9%
    仍失败                        16
  本来能过的                       11
    BROKEN（加计划反而挂）          9  = 82%
    both pass                       2

  按难度:  1→0/6   2→0/4   3→1/7
  按原始终止: task_completed→1/13  max_steps→0/3  repeated_truncation→0/1
  ```
  唯一救回的是 `6104387_3`（difficulty 3，65 步→47 步，计划点名了 `show_song_library`/`show_album_library` 的分页处理）。**净账：1 道失败变成功，9 道成功变失败，净 −8。**

### 12.1 与 §11 Gemini teacher 的配对比较（本节的主结论）

取两边 **control 都失败、且两边都评出了 repair 结果**的同一批题，n=15：

```
self-writer 救回 1/15 =  7%
gemini      救回 8/15 = 53%
只有 Gemini 救回: 7 道   只有自写手救回: 0 道
```

**严格占优**：Gemini 救回了自写手救回的那一道，外加 7 道，没有任何一道是自写手赢的。7 个不一致配对全部同向，符号检验 p ≈ 0.008，在 n=15 下已经显著。BROKEN 数同样是一边倒：自写手 9，Gemini 1。

**结论：guided replay 这个机制是有效的，失效的是"自己诊断自己"。** `appworld_sft_writer` 的 docstring 里那句刻意的设计选择——*"Same base model, no separately trained teacher -- this is a system prompt, not a fine-tuned component"*——正是瓶颈本身。模型在一道题上失败，恰恰说明它对这道题的理解有缺陷，再让它解释自己错在哪，得到的是同一个缺陷的二次表达。

自写手为什么会**主动有害**（82% 破坏率），机制上也说得通：计划是照着五周前那次失败写的，而那次失败今天已有 33% 不复现。writer 在诊断一个当下并不存在的错误，然后把 agent 推上一条为错误前提设计的流程。Gemini 只有 1 例破坏，说明更强的教师能识别出"这次其实没什么大错"。

### 12.2 顺带查出的两个问题

**(a) writer 会在 JSON 字符串里做思维链，烧光预算。** 5 道题（`29caf6f_1/2/3`、`692c77d_1`、`d0b1f43_2`）三次换 seed 重采样全部失败。抓到原始响应：`finish_reason=length`、`completion_tokens=4096`、内容 18649 字符，`plan` 字段正常写完（约 1321 字符），然后 `mistake_summary` 的字符串一开就再没合上，模型在里面自言自语 17k 字符直到耗尽预算。

根因：`enable_thinking=False` 关掉了推理通道，而 `response_format={"type":"json_object"}` 只约束"必须是合法 JSON"——**JSON 字符串内部可放任意长度文本，语法上没有任何收尾压力**，于是推理被写进了字段值。**加预算无用**（2048→4096 同样失败，只是啰嗦更久）。修法是换成带 `maxLength` 的真 JSON schema 让受约束解码机械地兜住（§11 的 Gemini 版本已经用了 `response_json_schema`，本地这条路反而是松的）。**尚未修。**

同一条路径在训练流水线里也存在且是静默的：memory writer 用同一个 `json_chat`，跑飞后 `run_router_chain` 把该题记为 `status: "error"` 并 `continue`——**这道题从梯度里整个消失**（不进 `live_decisions`），只留一行 error 日志。v5_nofeat 的 160 条记录里命中 1 条（0.6%），**也是 `29caf6f_1`**。

**(b) `base_train_v2` 的失败名单已严重过时。** 33 道里今天有 11 道（33%）无计划就通过；§11 的 control 独立测出 9 道（27%）。两次 control 差 2 道，量级与 §13 记录的跑间不确定性一致。成因仍三者混淆：08-28 至今的配置漂移、当天把 agent `max_tokens` 2048→4096、以及跑间不确定性。**影响面超出 SFT 这条线：今天之前测的所有基线都需要重新核对**，包括 `baseline_recheck` 的 0.4912 和 §7 的 G=8 0.5789（那两个数彼此同条件、结论仍自洽，但今后任何 4096 下的新数字都不能跟它们比）。

### 12.3 必须标注的限制

**k=1、temperature=0，是下界。** rejection-sampling SFT（STaR/RFT）这类方法的标准做法是每题采 k=4~16，靠分布尾部捞正例；温度为 0 且只采一次，等于只取了分布上的一个确定性点。n=17 下 5.9% 的 95% 置信区间约 0.1%~28%——能排除"三成以上"，分不清 5% 和 20%。**注意：正因为 k=1，"重试"必须改变 seed 或计划，否则同 seed 同温度会逐 token 复现同一条轨迹。** 训练循环目前没有任何重试：`replay_and_verify` 一次不中即放弃，且 replay seed 只含 `batch_idx`、不含 iteration，所以跨 iteration 的"第二次机会"在同一候选槽位上是空转。

---

*后续每次新跑动，在此文件末尾追加一节，格式同上：reward 定义、关键参数、产物目录、结果。*

## 13. `train_router_llm_grpo.py`（router 换成 LoRA 可训练的 LLM，本机 COE-CS-sv002）

- **改动**：router 从 144 参数线性分类器（`router_policy.RouterPolicy`）换成 **Qwen3.5-35B-A3B + LoRA**
  （`src/trajectory_memory_lab/router_llm_trainable.py`）。reward 定义、batch 切分、GRPO advantage、
  "随机选一个候选延续 bank"这些**全部沿用 §9 的 `train_router_selfreward.py`，一行没改**——换的只是
  "谁做决定"和"梯度打在哪"。`train_router_selfreward.py` 本身没有被修改，两个脚本是并列实验。
  详细设计见 DESIGN.md §16。
- **reward**：与 §9 完全相同。`advantage_k = self_pass_rate_k − mean_j(self_pass_rate_j)`，
  用本 batch 自己的 10 道题 replay 打分；`self_baseline` 只记日志、不进梯度（§14.1 的结论沿用，
  没有重新引入 K+1 均值）。
- **动作参数化（这是本节的核心）**：不是自由生成。强制 assistant 轮以 `{"route": "` 开头，四个 route
  的**首 token 两两不同**（memory=17269 / s=82 / both=21028 / ne=811，构造时对真实 tokenizer 断言），
  所以一次前向在这一个位置上对 4 个 logit 做 `log_softmax` 就是完整策略——可微、熵精确，和线性 router
  的 `Categorical` 是同一个东西。
  **因此没有用 vLLM，也没有用 LoRA 热加载**：两者在本仓库 vLLM 0.17.1 上都确实可用（已查证
  `POST /v1/load_lora_adapter` + `load_inplace`，以及 chat-completions 的 `logprobs`），但 router 每个
  决策只要 ~0.2s，一个 batch 80 个决策 ≈ 1-2 分钟，对比同 batch 的 AppWorld self-eval ~5.8 小时，
  推理引擎没有可优化的东西。采样和梯度共用同一份权重，所以重算的 logprob **就是**采样时那个值
  （实测 `|diff| = 0.00e+00`），也没有任何东西需要在 `optimizer.step()` 之后同步。
- **关键参数**：Qwen3.5-35B-A3B（LoRA r=8, alpha=32, all-linear, **11,238,720** 个可训练参数——比
  8B 的 21.8M 少，因为 hidden_size 只有 2048，而 256 个专家是融合的 3D `nn.Parameter`
  （`mlp.experts.gate_up_proj`）不是 `nn.Linear`，`all-linear` 够不到它们；LoRA 落在
  linear_attn / self_attn 投影、shared_expert 和 MoE 门控上），
  **lr=1e-5**（不是 §9 的 0.01，也不是本仓库 SFT 的 1e-4——实测 1e-4 一步就把某个 route 概率从
  0.00000 推到 0.949 且熵正则发散，见 DESIGN.md §16.4），`--max-grad-norm 1.0`，
  `entropy_coef=0.01`（沿用 v4 标定），`policy_temperature=1.0`，K=8，batch_size=10。
- **GPU / 端口 / 隔离（本机与另一台机器共享 NFS，全部避开）**：
  - 两个确定性 TP=2 副本：**8030（GPU 0,1）/ 8031（GPU 4,5）**，tmux `routerllm_server_a` /
    `routerllm_server_b`，日志 `router_reward_v1/router_llm_grpo_v1_server_{a,b}.log`。
  - router 训练副本在 **GPU 6,7**（Qwen3.5-35B-A3B，`device_map="auto"`，权重 32.3GiB/卡）。
    6 张空闲卡全部用上，没有余量。
  - **不在本仓库 `.venv` 里跑**：`.venv` 装的是 vLLM 0.17.1，它钉 transformers 4.57.3，而 4.57.3
    不认识 `qwen3_5_moe`（`AutoConfig` 直接拒绝这个 checkpoint）。router 走单独的
    **`/nas04/yixuh/router_venv`**（transformers 5.17.0 + peft 0.21.0 + torch 2.10.0+cu128），
    两个 serving 副本的环境一点没动。`AutoModelForCausalLM` 映射到 `Qwen3_5MoeForCausalLM`，
    只建语言塔（等价于 server 的 `--language-model-only`），71.9GB checkpoint 里加载约 64.6GB。
  - **GPU 2、3 和端口 8001/8002 是别人的**（用户 `haskari` 的 `judge_server.py`，已跑 18 天），没有动。
  - **`APPWORLD_ROOT=/nas04/yixuh/appworld_root_llmgrpo`**（`data/` 拷了一份 194M，
    `experiments/outputs` 全新空目录），按 §10 的教训隔离，不与 v5 / v5_nofeat 抢同一批 task 目录。
- **产物目录**：`router_reward_v1/router_llm_grpo_v1/`（冒烟批次写在
  `router_reward_v1/router_llm_grpo_v1_probe1/`，不污染正式目录）。
- **开工前的测量（DESIGN.md §16.3，记在这里因为它差点得出相反结论）**：GRPO 要求 K 个候选真的会做出
  不同决策，否则 advantage 全 0、梯度全 0，而日志看起来完全正常。第一次用 `cheap_train_v5` 的真实
  record 量，测出 8B 平均熵只有 0.037 nats、`sft`/`both` 概率≈0，像是彻底坍缩——**但那 20 条 record
  里 0 条同时带 memory 和 sft 草稿**，`sft`/`both` 本来就没东西可提交，给 0 概率是正确判断。补上真实
  Gemini 修复计划让四个 route 都可选之后重测：8B 在 T=1.0 下平均熵 0.154、p(argmax)=0.931、
  **P(8 个候选完全相同)=0.003**，探索足够。顺带一条反直觉结论：**模型越大越确定、探索越少**
  （8B 的熵只有 1.7B 的一半）。
- **冒烟测试**（三个，全部在真实训练启动前跑过并通过，见 DESIGN.md §16.5）：
  `smoke_router_llm_logprob.py`、`smoke_router_llm_policy_gradient.py`、
  `smoke_router_llm_batch_e2e.py`。
- **⚠️ `--enable-sft-lora` 默认关闭，开之前必须先导出服务端变量**：`scripts/router_sft_lora_update.sh`
  的默认值是 `det_server_a`/`det_server_b` + 端口 8000/8001 + GPU 0,1/2,3——**跟本机的实际配置全不一样**
  （本机是 `routerllm_server_a`/`routerllm_server_b` + 8030/8031 + GPU 0,1/4,5），而且 GPU 2,3 是
  `haskari` 的。照默认值跑会去 kill 不存在的 session（脚本会 abort，这是好的），但如果 session 名碰巧
  对上了就会拿别人的卡去训练。要开这条路必须先
  `export SERVER_A_NAME=routerllm_server_a SERVER_B_NAME=routerllm_server_b SERVER_A_PORT=8030
  SERVER_B_PORT=8031 SERVER_A_GPUS=0,1 SERVER_B_GPUS=4,5`。这就是 §10 那条教训的复现，所以这一版把
  LoRA 重训做成了显式开关而不是默认行为——第一次跑不应该有能力把两个副本弄下线。
  guided replay 本身照常跑（只有"攒够 8 条就重训"这一步被 gate 住），所以 sft 这条路的产出率照样能观测。
- **预算（`--rollouts-per-batch 8 --batch-size 10` 的实测估计）**：~56.6h/iteration =
  8.0h 起草（720 次 × ~40s，实测值，串行只用一个副本）+ 42.0h self-eval（4 轮 × 9 batch × 70min）
  + 2.1h sft guided-replay（未验证）+ 4.5h validation。
  **注意：§9 给 v5 报的 49.6h 漏算了起草时间**——`run_router_chain` 是每个候选都要把这 10 道题全部起草
  一遍（§14.3 的 content-before-route），也就是每个 batch `K × batch_size = 80` 次起草而不是 10 次，
  而且它串行占用一个副本、不与评测重叠。v5 的真实耗时应该也要在它的估计上加 ~8h。
- **换成 MoE 之后必须补的三件事（都不是可选项，全部实测过）**：
  1. **梯度检查点是"能不能跑"而不是"快不快"**：35B 权重占 32.3GiB/卡，47.4GiB 可用里只剩 15GiB 给
     激活。不开检查点时 **1,024 token 的反向峰值就有 40.6GiB**（单个决策 8.3GiB 激活），4,096 token
     直接 OOM——而 router 提示中位数是 11.5k。开了之后 1,024 降到 33.0GiB，16,384 能过（42.8GiB），
     24,576 仍 OOM。代价是更新变慢：8k 时 19.2s、16k 时 64.4s。
     **坑**：transformers 只在 `if self.gradient_checkpointing and self.training` 时走检查点分支，
     而 router 是刻意 `eval()` 的，所以第一次开完 `is_gradient_checkpointing=True` 但峰值一点没变。
     现在带梯度的重算包在 `train()` 里（`TrainableLLMRouter._checkpointing`）。**这不改策略**：
     该模型 `attention_dropout=0.0`、`lora_dropout=0.0`，两种模式下四个 route logit
     **逐位相同（max |diff| = 0.0）**，实测过而不是假设——这正是
     `verify_recompute_matches_sample` 仍然能断言 `|diff| = 0.00e+00` 的前提。
  2. **router payload 必须截断**（这条与 MoE 无关，8B 也会撞）：今天 11:34 改过
     `router_llm_policy.build_router_payload`，把**完整**对话记录放进了 payload，而 `probe1`（10:33）
     跑的是改动之前的版本。结果 90 条训练轨迹里 **66 条**光 payload 就超过原来的
     `max_prompt_tokens=8192`（中位数 11,536，最长 30,713）。**砍草稿没用**：起草的 memory 候选中位数
     只有 184 token、sft 计划 285 token，两者合计约占提示的 3%，而且正是 §14.3 让 router 判断的东西。
     所以只截断对话记录：`ROUTER_MAX_STEP_CHARS=31_000`，保留头尾完整的 step、**尾部优先**（AppWorld
     的失败出现在结尾，开头多是任务铺垫），中间插一条显式的省略标记，并在 payload 里加
     `trajectory_steps_elided` 计数——让 router 能区分"agent 这段什么都没做"和"这段被截掉了"。
     按字符而不是 token 计预算，是因为这个函数同时服务于没有 tokenizer 的 `decide_route`，而
     `router_llm_trainable` 的"未训练探针即 step-0 基线"论证依赖两边看到逐字节相同的文本；
     2.646 chars/token 是 90 条里最低的比值，所以字符预算换算出的 token 上界对每一条都成立。
     截断后实测（配最坏情况草稿）：**90/90 全部落在 16,384 守卫之下**，最长 12,768、中位数 10,261，
     55/90 的轨迹被省略了中段。
  3. **新 venv 少装 `google-genai`，把 `sft`/`both` 两条路整个弄死了，而日志完全正常**：
     `appworld_sft_writer.generate_plan_with_teacher` 的第一个失败分支就是
     `from google import genai` 的 `ImportError -> return None`，而它按设计对**任何**失败都静默返回
     `None`（"Gemini 短暂不可达不该让整个决策失败"）。`router_venv` 是为了绕开 transformers 版本冲突
     新建的，只装了 torch/transformers/peft/openai，于是 20 条 record **全部**拿不到修复计划。
     这正是 §16.3 那个陷阱的原样复现：没有计划可提交时，`sft`/`both` 概率低是**正确判断**，
     照着读会得出"策略没塌"的结论，而实际上 4 分类动作空间已经塌成 2 分类。
     **能发现它，是因为这次在 record 里加了 `drafted_sft_plan_available` 这个直接诊断位**
     （`router_bank_builder`）——原有的 `sft_status` 是从**选中的 route** 反推的，回答不了
     "当时到底有没有计划可选"。装上 `google-genai==1.56.0` 后实测 3/3 出计划，探索度表已重测。
  4. **三个冒烟测试在 35B 上全部重跑并通过**，`|diff| = 0.00e+00` 在检查点路径下依然成立，
     梯度只进 LoRA（350/700 个适配器张量拿到非零梯度）、基座零梯度。lr 扫描重测的结论与 §16.4 同向，
     `lr=1e-5` 不用改：一步 +1.52e-02、9 步到 p=0.683（8B 当时 9 步到 ~0.87），5e-5 起一步就饱和到 1.0。
- **探索度重测（35B，n=20 真实 record，13/20 带真实 Gemini 计划，`scripts/probe_router_llm_entropy.py`）**：

  | 模型 | T | 平均熵 (nats) | p(argmax) | P(8 候选单决策全同) | 平均 route 概率 | argmax 分布 |
  |---|---|---|---|---|---|---|
  | Qwen3-8B（§16.3 原表） | 1.0 | 0.154 | 0.931 | 0.0033 | mem .177 / sft .000 / both .136 / neither .686 | neither 13, memory 4, both 3 |
  | **Qwen3.5-35B-A3B** | **1.0** | **1.171** | **0.473** | **0.0119** | mem .268 / **sft .263** / both .136 / neither .333 | memory 8, neither 8, sft 3, both 1 |

  **§16.3 那条"模型越大越确定、探索越少"的反直觉结论，在这里被推翻了**：35B MoE 的熵是 8B 的 7.6 倍，
  已经接近均匀（ln4=1.386），整个 batch 全同的概率 ~0。温度不用动，`policy_temperature=1.0` 保持。
  更重要的是，§16.3 留的那个**已知缺口关上了**——8B 在 T=1.0 下几乎不单独选 `sft`（p≈0.000），
  纯 `sft` 这个动作是半死的；35B 给到 p(sft)=0.263，argmax 里也真的选了 3 次，四个 route 都活着。
  （这张表的第一版测出 p(sft)=0.047 且 `drafted_sft_plan_available` 为 0/20，就是上面第 3 条那个
  缺依赖的坑，已作废重测。）
- **实测速率**：起草 ~27s/任务（比预算假设的 40s 快）；router 自身前反向在 ~11k token 的中位提示上
  约 2.7s 采样 + ~35s 反向，一个 batch 80 个决策约 50 分钟，约 1.2h→7.6h/iteration，
  仍是 self-eval 42h 的小头，但不再是 8B 时那个"1-2 分钟/batch"可忽略量。
- **状态**：2026-09-18 20:15 正式启动，tmux `train_router_llm_grpo`，日志
  `router_reward_v1/train_router_llm_grpo.log`。预算 ~56.6h/iteration（不含 router 自身的 ~7.6h）。
  冒烟批次（K=2、1 个 batch、真实 self-eval）此前已完成于 `probe1`，但跑的是 8B + 截断前的 payload，
  不能直接当作本次配置的验证。

## 14. `run_router_llm_probe.py`（LLM router 不训练，本机，已完成，与 §13 互补）

- **目的**：§13 是真正 GRPO 训练 LLM router 的权重；这节是同一个 `router_llm_policy.py`（同一套 prompt、
  同一个 payload 构造）在**完全不训练**的情况下，纯靠 `qwen35-tau`（跟 task agent 同一个 35B MoE，
  temperature=0）自己的判断跑一遍，看 step-0 的路由分布和下游效果长什么样——这也是 §13 训练如果真的
  有进展，将来用来对比的起点。
- **模型**：没有用单独的小模型（最早试过 Qwen3-1.7B 单独起服务，用户否决，要求跟 task agent 同型号），
  直接复用本机 `det_server_a/b`（`qwen35-tau`），不需要额外显存/端口。
- **输入**：`router_llm_policy.build_router_payload` 现在包含任务的完整 instruction + 完整
  trajectory_steps（不再只是 success/reward/termination_reason 的摘要），加上 active_memory_count、
  最近两个 batch 的 bank 变动、已经起草好的 memory/sft 候选内容——细节和截断逻辑见 §13 的记录（两边共用
  同一个 `build_router_payload`）。
- **产物目录**：`router_reward_v1/router_llm_probe_v1/`，日志 `router_reward_v1/router_llm_probe.log`。
- **结果（90/90 路由决策 + 真实 57 题 dev 评测）**：
  ```
  route_counts = {'both': 11, 'memory': 16, 'neither': 31, 'sft': 32}，active_entries=14
  成功题 → routes: {'both':10, 'memory':15, 'neither':10, 'sft':22}
  失败题 → routes: {'neither':21, 'sft':10, 'memory':1, 'both':1}
  ```
  **完全没有坍塌**——四个 route 都在用，且方向合理：失败题几乎不写 memory/both（1/33），
  多数走 neither 或 sft；成功题里 sft（巩固计划）反而最常见。
- **⚠️ 首次算出的 0.4737 一度误判**：dev 评测默认走 `run_appworld_rollout.py` 现在的 `max_tokens=4096`
  默认值，而能直接比的 `baseline_recheck`(0.4912)/G=8(0.5789) 都是旧的 2048 配置下测的——§12.2(b) 已经
  指出这条不可比。当场在同一个副本（127.0.0.1:8000）、同 seed、同 4096 配置补跑了一次无记忆基线：
  ```
  router_reward_v1/baseline_recheck_4096/summary.json: pass_rate = 0.4211（24/57）
  ```
  **真正可比的结果：LLM router 0.4737 vs 无记忆基线 0.4211，+5.26pp。**
  一个完全没训练、只是能读到真实任务内容和完整轨迹的 35B router，判断力已经比不给任何记忆更好——
  这是 §13 GRPO 训练如果有效，需要超过的起点。

## 15. ALFWorld 兼容层（本机，第二个 benchmark，已跑通端到端）

- **动机**：用户要求找一个"不那么偏应用"的第二个 benchmark 跑同一套流程（明确否决了 tau2-bench——
  "感觉tau bench太偏向非常实际的应用了"）。选了 ALFWorld：TextWorld 文字冒险式的家务任务，
  纯文本模式不需要 Docker/显示/GPU 渲染，跟 AppWorld（代码执行沙盒）在任务性质上足够不同。
- **Python 版本坑**：`textworld==1.7.0` 的 `EvalSymbol.derive()` 用
  `locals().update(context["variables"]); eval(self.expression)` 这种 CPython 反模式，在 3.11+ 的
  局部变量优化下会报 `NameError`（3.13 上实测复现）。ALFWorld 官方本来就要求 3.9/3.10，不是环境配置错，
  是库本身不兼容新 CPython。**解法**：单独建了 `/nas04/yixuh/alfworld_venv310`（Python 3.10，
  `python3.10 -m venv --without-pip` + 手动 `get-pip.py` 引导，因为原生 venv 没 `ensurepip`、
  `virtualenv --download` 会在 NFS 上卡死），装了 `alfworld==0.4.2`、`pyyaml`、`openai`，
  再 `pip install -e . --no-deps` 把本仓库的 `trajectory_memory_lab` 包也装进这个 venv
  （`playwright` 没装但这条路径用不到，忽略那条 pip 冲突警告）。
- **数据**：`alfworld-download` 到 `/nas04/yixuh/alfworld_data`（2.3GB），
  `json_2.1.1/{train,valid_seen,valid_unseen}` 三个目录，配置写在
  `/nas04/yixuh/alfworld_data/base_config.yaml`。
- **训练/测试分离（用户明确要求"记得训练集和测试集要分清楚"）**：`train`(3553题)→写记忆/SFT 草稿的
  基础 rollout 池，等价于 `base_train_v2`；`valid_seen`(140题)="eval_in_distribution"，同房型、
  未见过的具体任务组合，较弱的分布内检验；`valid_unseen`(134题)="eval_out_of_distribution"，
  房间布局本身就没在 train 里出现过，是真正的强泛化测试，应作为主 eval 指标。
  用 `list_available_tasks()` 直接对三个目录各自扫描一遍并集合求交验证：
  **train/valid_seen/valid_unseen 两两 task_id 交集均为 0**——没有任何题目跨 split 出现。
- **新文件**：
  - `src/trajectory_memory_lab/alfworld_agent.py`：`AGENT_SYSTEM`（文字指令：每轮从
    admissible_commands 里原样回复一条，不接受的话最多重试 2 次后判 `ungrounded_action` 终止）、
    `extract_command()`（精确/宽松匹配 admissible 列表）、`list_available_tasks()`（复现
    `AlfredTWEnv.collect_game_files` 的过滤逻辑：solvable、6 种已知 task_type、排除 movable/Sliced）、
    `load_task_env()`（单任务单 env：先让 `AlfredTWEnv.__init__` 走完整个 split 的
    `collect_game_files`，再把 `game_files` 收窄成一个再 `init_env`——没做进一步优化，
    每次单任务 rollout 都要付几秒钟全 split 扫描的代价）、`run_task()`（驱动一整局，
    产出跟 AppWorld 同形的 canonical trajectory：`{"source_task_id","domain":"alfworld",
    "task":{"id","instruction"},"success","reward","termination_reason","evaluation","steps"}`，
    role 映射比 AppWorld 少一种——没有单独的 tool role，环境反馈和初始房间描述都算 `user`）。
  - `scripts/run_alfworld_rollout.py`：结构照抄 `run_appworld_rollout.py`（每题一个子进程、
    `--split train/valid_seen/valid_unseen`、`--memory-bank`/`--memory-top-k` 复用
    `memory_retrieval.retrieved_block`——这个函数本来就跟 benchmark 无关，直接可用）。
    必须用 `alfworld_venv310` 的解释器跑，不是仓库默认 `.venv`。
- **端到端冒烟测试（真实调用 det_server_a，qwen35-tau，temperature=0，1 题，`--split train`）**：
  跑通，无崩溃，产出合法 trajectory（41 步，`termination_reason=max_steps`，未成功）。
  读了完整 transcript：agent 找到了目标 desk 上的 alarmclock，但下一步试了
  `examine alarmclock 1`（不在 admissible 里，桌上物体不能直接 examine），收到"不是合法命令"的提示后
  没有改试 `take alarmclock 1 from desk 1`，反而反复回复 `look`，40 步内没能恢复——这是 agent 在
  未给记忆时的真实探索能力问题（prompt/策略层面），不是 harness 的 bug，留待后续用记忆/SFT 改善，
  这次的目的只是验证技术管线本身。
- **尚未做**：没有跑过 base train rollout 全池（对齐 `base_train_v2` 那种"先批量收集基础轨迹"的规模），
  没有把 router/memory/SFT 全流程接到 ALFWorld 上——目前只验证了环境层 + agent harness + rollout
  runner 三者能跑通、split 分离正确。

- **第一批真实无记忆 rollout（2026-09-19，本机，已完成，0 错误）**：
  两个批次分别打到 det_server_a（8000）/det_server_b（8001），4-way client 并发（server 端
  `--max-num-seqs 1` 强制串行，client 并发不提速，只是不添乱），跑完各自才发现两边 tmux 会话已自然退出：
  - `alfworld_experiment/base_train_v1/`：train split 随机抽样 40 题，**pass_rate = 0.825**（33/40，
    6 超步数上限、1 `ungrounded_action`）——这是未来喂给 router/memory/SFT 起草的基础轨迹池，
    等价于 `base_train_v2`。
  - `alfworld_experiment/baseline_valid_unseen_v1/`：valid_unseen split（真正的分布外测试集，
    房间布局在 train 里完全没出现过）随机抽样 57 题，**pass_rate = 0.7544**（43/57，13 超步数、
    1 `ungrounded_action`）——**这是要拿来跟未来记忆/SFT/router 方案对比的无记忆基线数字**，
    跟 AppWorld 那边"先测 baseline 再看增量"的方法论一致；后续任何"记忆有没有用"的结论，都必须是
    同一批 57 题、同一份 `base_config.yaml`、同一个 replica 上测出来的数字，不能跨配置比较
    （AppWorld 那边 §7/§14 已经因为跨配置比较踩过两次坑）。
  - 40 题里唯一的 `ungrounded_action` 和 57 题里唯一的 `ungrounded_action` 都是模型连续 3 轮给出不在
    admissible_commands 里的动作被硬终止——真实的 agent 能力问题，跟 harness 无关，样本量还小，暂不
    单独分析。

- **router/memory 接入 ALFWorld，第一次端到端跑通（2026-09-19，本机，已完成）**：
  - **代码改动**：`RouterBuilderConfig` 新增 `domain` 字段（默认 `"appworld"`），`domain="alfworld"` 时
    跳过 sft 计划起草（还没有 ALFWorld 版教师 prompt / guided replay 脚本，硬套 AppWorld 的
    `apis.spotify.login` 式 prompt 只会让模型幻觉出根本不存在的 API，比不写计划更糟）；
    `writer_rubrics.routed_writer_system()` 新增 `domain_description` 参数，让写手 prompt 的开场白从
    "customer-service agent" 换成 "a household-task agent operating in a text-adventure environment"。
  - 新文件 `scripts/run_alfworld_router_llm_probe.py`，照抄 `run_router_llm_probe.py` 的结构：
    对 `base_train_v1` 的 40 题跑 `router_mode="llm"` 建 bank，再用同一批 57 题 valid_unseen 样本
    （跟 `baseline_valid_unseen_v1` 完全同一份 task_ids）测 pass_rate，保证可比。
  - **路由分布（40 题）**：`{'neither': 28, 'memory': 12}`，sft/both 均为 0（预期内，域内未接 sft）。
  - **⚠️ 真实发现：写手过度 refine，12 次 memory 写入最后只剩 1 条活跃记忆**——从 position 26 起，
    写手几乎每次都选 `refine` 指向当前唯一的活跃条目，即使话题重叠度很低（`dup_best_overlap` 低至
    0.11-0.17，明显跟上一条不是一回事）也照样 refine，每次只保留上一条 20%-47% 的内容
    （`refine_retention`），链式合并 9 次后从 9 条 add/refine 记录坍缩成 1 条幸存条目。这正是
    模块注释里点名过的"v4 比不写记忆还差"那种失败模式，只是这次不是旧的强制 dedup 规则造成的，
    是写手自己的判断倾向——猜测根因是 ALFWorld 这批任务类型窄（heat/cool/take/clean 几种程序性
    校验经验），写手把"都是操作校验类经验"当成同一话题，倾向塞进同一条而不是像 AppWorld 那样
    有更多不同 app/API 话题天然撑开话题空间。**尚未修复，只是测出来了**——用户明确要求先跑通全量
    看 pipeline 能不能跑完，不要停下来纠结这个发现，所以先如实记录，修复留待下一步。
  - **57 题 valid_unseen 对比结果**：`alfworld_experiment/router_llm_probe_v1/eval_valid_unseen57/
    summary.json`: pass_rate = **0.7544（43/57）**，跟无记忆基线 `baseline_valid_unseen_v1` 的
    0.7544（43/57）**总数完全打平**——但逐题对比不是巧合性的"完全没变化"：57 题里有 6 题结果不同
    （3 题从失败变成功，3 题从成功变失败，净抵消为 0），且这 6 题里 5 题是 `pick_heat_then_place_in_recep`
    / `pick_cool_then_place_in_recep`，跟那条幸存记忆的内容（"heat 类指令执行前先验证...")主题高度
    相关——说明这条被过度合并、信息严重损耗的记忆确实在起作用，只是在这 57 题的小样本上恰好正负相消，
    不能说"记忆完全没用"，也不能说"记忆有正向增量"，样本噪声下暂时不可分辨。
  - **结论/下一步**：ALFWorld 上 router+memory 的技术管线（起草→路由→提交→检索→复用）第一次完整跑通，
    产出了一个跟基线严格可比的数字。但 refine 过度合并的问题是真实的，如果不修，扩大训练池规模只会让
    这条"幸存记忆"越来越通用、越来越信息稀薄——下一步要么调整 `routed_writer_system` 里对 refine 的
    门槛提示，要么扩大训练池的任务类型多样性再看这个倾向是否是小样本假象，两个方向都还没做。

- **补上 ALFWorld 的 sft 教师模型 + guided replay（2026-09-19，本机，已完成）**：
  用户明确要求把上面跳过的 sft 路径补齐，而不是长期只有 memory 能用。
  - **新文件 `alfworld_sft_writer.py`**：ALFWorld 版的自写手 prompt（`ALFWORLD_SFT_WRITER_SYSTEM`）和
    教师 prompt（`GEMINI_TEACHER_SYSTEM`），针对 admissible-command 动作空间重写措辞（"go to fridge 1"
    /"heat mug 1 with microwave 1" 这类，不再提 apis.*）；`build_writer_payload`/`validate_writer_output`
    /`generate_plan_with_teacher` 的 Gemini 调用/重试/校验逻辑全部复用 `appworld_sft_writer.py`（给
    `generate_plan_with_teacher` 加了 `system_prompt` 参数，默认值是 AppWorld 的 prompt，不影响现有调用），
    没有重复实现。
  - **新文件 `scripts/run_alfworld_guided_replay.py`**：照抄 `run_appworld_guided_replay.py` 的结构，
    区别是要多传 `--split`（ALFWorld 的 task_id 只在单个 split 目录内唯一，不像 AppWorld 能直接查全库）。
  - **`router_bank_builder.py`**：删掉了"domain!=appworld 就跳过 sft 起草"的临时限制，改成按
    `config.domain` 动态 `importlib.import_module` 出 `appworld_sft_writer` 或 `alfworld_sft_writer`，
    self/teacher 两条路径都走同一份逻辑，只是模块换了。
  - **`router_sft_pipeline.py`** 的 `replay_and_verify`/`collect_batch_sft_examples` 加了 `domain` 参数
    （通过 `_REPLAY_CONFIG` 字典选解释器/脚本路径/env 变量/agent prompt），默认值还是 AppWorld 的老配置，
    `train_router_selfreward.py` 的调用点完全不用改。
  - **`run_alfworld_router_llm_probe.py`** 加了 `--sft-writer`（teacher/self）和 sft 候选的
    replay+verify 步骤，复用刚才泛化的 `router_sft_pipeline` 函数。
  - **第一次真实结果（40 题 train，`sft_writer=teacher`）**：
    ```
    route_counts={'neither': 29, 'sft': 10, 'memory': 1}
    sft/both committed: 10 candidate(s) to replay+verify
    sft replay: 10 replayed, 9 verified success (yield=0.9)
    ```
    有 sft 选项可用之后，router 几乎不再选 memory 了（只剩 1 条，之前那批 12 条里大部分现在改选 sft）——
    这顺带让上面记录的"refine 过度合并"问题的实际影响面变小了很多（这次只有 1 条 memory 写入，没有
    连续 refine 的机会）。sft 候选的复核成功率 0.9（9/10）看起来比 AppWorld 那边教师模型的 0.625 高，
    但**不是同一件事**：这 10 个 sft 选择里 8 个来自 base_agent 本来就成功的任务（巩固计划，重放时任务
    本来就不难），只有 2 个来自失败任务（真正的修复计划）——巩固计划复核成功率天然应该远高于修复计划，
    不能直接拿 0.9 跟 AppWorld 的 0.625（那边测的是纯失败任务的修复率）比较。真正可比的"修复率"要单独
    看这 2 个失败任务的样本，样本量太小还看不出结论。
  - **57 题 valid_unseen 对比评测（已完成）**：`alfworld_experiment/router_llm_probe_v2/
    eval_valid_unseen57/summary.json`: pass_rate = **0.7193（41/57）**，比无记忆基线 0.7544（43/57）
    低 3.51pp（41 vs 43，差 2 题）。这个评测测的仍然只是 memory bank 的效果（sft 产出进的是训练池，
    不影响这次 retrieval-only 的评测）——这次 bank 里同样只有 1 条活跃记忆（sft 分流走了大部分本来会
    进 memory 的任务，剩的这一条内容也跟 v1 那条不同），跟 v1 探针"1 条记忆、57 题打平"的结果对照看，
    差 2 题的量级符合单条记忆在这个样本量下的正常噪声范围，暂时不能下"记忆拖累了效果"的结论，也不能
    说"没用"——目前样本太小、bank 太薄，还看不出信号。
  - **现状小结**：ALFWorld 的 sft 教师模型 + guided replay 已经补齐并跑通（10 replayed → 9 verified，
    yield=0.9，但 8/10 是巩固计划不是修复计划，不能直接跟 AppWorld 的修复率 0.625 比）；memory 这条线
    因为 sft 分流，这一批只写了 1 条，还没有观察到有意义的正向或负向信号。下一步如果要看清 memory 到底
    有没有用，需要更大的训练池规模（不止 40 题）才能攒够足够多条独立记忆来看整体效果，而不是被 1-2
    条记忆的偶然内容主导结论。
