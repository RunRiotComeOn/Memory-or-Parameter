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

## 6. `train_router_selfreward.py` + 熵正则（v4，另一台机器在跑）

- **改动**：在 loss 里加熵正则项 `loss = pg_term − entropy_coef × entropy_sum`（`entropy_coef` 默认 0.01），直接鼓励 route 分布不要过早收敛成确定性策略；同时把 `lr` 从 0.05 降到 0.01。验证阶段（贪心）也顺手记录 `mean_entropy`，即使 argmax 还没变、也能提前看到底层分布在不在塌。
- **动机**：v2、v3(G=4) 两次独立训练（不同 reward 定义）都坍塌成确定性策略，且实测 v3(G=4) 的采样熵从 1.14 nats 掉到 0.65 nats（4 分类最大熵 ln4≈1.386）——判定是训练动力学问题（无正则、更新次数少、线性模型容易饱和），不是换 reward 定义能单独解决的。
- **产物目录**：`router_reward_v1/cheap_train_v4/`（另一台机器写，共享此 NFS 目录）
- **状态**：进行中，由用户在另一台服务器上运行，我不管理这个进程的生命周期。

---

*后续每次新跑动，在此文件末尾追加一节，格式同上：reward 定义、关键参数、产物目录、结果。*
