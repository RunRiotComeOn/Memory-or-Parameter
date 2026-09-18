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

---

*后续每次新跑动，在此文件末尾追加一节，格式同上：reward 定义、关键参数、产物目录、结果。*
