<!-- Language links are GitHub-compatible navigation links. GitHub Markdown does not execute JavaScript, so the HTML companion provides true toggle behavior. -->

<p align="center"><a href="#中文">🇨🇳 中文</a>　|　<a href="#english">🇬🇧 English</a></p>

---

<a id="中文"></a>

# Fast-WAM：World Action Models 是否需要在推理时想象未来？

> **论文**：[arXiv:2603.16666](https://arxiv.org/abs/2603.16666)（v2，2026-03-23）｜**作者**：Tianyuan Yuan 等｜**机构**：清华大学 IIIS、Galaxea AI  
> **一句话结论：** WAM 的主要收益可能来自“训练时学会预测未来”，而不是“每次执行动作前都生成未来视频”。Fast-WAM 保留视频联合训练，推理时只编码当前观测，再生成动作。

[中文](#中文)　|　[English](#english)　|　[论文](https://arxiv.org/abs/2603.16666)　|　[代码](https://github.com/yuantianyuan01/FastWAM)

## 1. 要解决什么问题？

传统 WAM 通常采用 **imagine-then-execute（先想象、再执行）**：先根据当前图像和任务指令生成未来视频，再根据想象出的未来状态预测机器人动作。视频需要多步去噪，推理延迟较高。作者提出一个关键问题：性能提升究竟来自训练时的视频预测监督，还是来自推理时显式生成未来视频？此前两者通常同时存在，难以区分。

## 2. 已有方法：范式、代表方法与取舍

| 范式 | 具体做法／代表 | 优点 | 主要代价 |
|---|---|---|---|
| VLA 直接策略 | OpenVLA、π₀/π₀.₅：图像＋语言→动作 | 路径直接；可利用视觉语言预训练的语义知识 | 通常不以未来视频预测作为核心训练目标，物理动态通过策略隐式学习 |
| WAM 联合生成 | Unified World Models、Motus 等：未来视频 Tokens 与动作 Tokens 联合去噪 | 视频和动作在同一生成过程中建模，可共享时序／交互信息 | 推理需处理视频与动作两类生成变量，计算量较大 |
| WAM 先视频后动作 | Vidar、Causal World Modeling、LingBot-VA 等相关路线：先生成未来观测，再由 IDM／动作头预测动作 | 动作显式依赖预测的未来状态 | 视频生成成为动作执行前的额外阶段，延迟较高 |
| Fast-WAM | 训练保留视频协同目标；推理删除未来视频分支 | 保留视频监督，同时避免在线生成未来视频 | 不再向动作模块提供显式生成的未来画面 |

> 注：论文将现有 WAM 归纳为两种主要结构（Fig.1）：**联合去噪**与**先视频后动作**。上表中的论文名称是相关路线代表，不意味着论文对每个模型都进行了同条件复现比较。

## 3. 本文方法与基线差异

Fast-WAM 的关键变化是**推理路径**，不是简单换一个更小的视频模型：用预训练 Wan2.2-5B 视频 DiT 作为世界表征编码器；训练时仍预测未来视频 Latent；推理时只输入当前帧的干净 Latent，视频 DiT 单次前向提取世界表征，再由 Action Expert 生成动作。

论文构造两个受控基线（Sec.3.3）：
- **Fast-WAM-Joint**：视频 Tokens 与动作 Tokens 在共享模型中联合去噪，动作生成与未来视频生成耦合。
- **Fast-WAM-IDM**：先生成未来视频 Tokens，再把未来表征交给动作预测模块。
- **Fast-WAM w.o. video co-train**：架构和推理不变，只移除训练时的视频预测损失，用来检验视频协同训练本身的作用。

## 4. 输入、输出与数据如何表示？

- **输入**：当前多相机 RGB 观测 \(o\)＋自然语言任务 \(l\)。多相机图像先拼接成一张图，再输入预训练 Wan2.2 VAE。
- **视觉 Tokens**：VAE 将图像／视频压缩到连续 Latent 空间；当前首帧为干净视觉 Tokens，未来帧经 VAE 得到未来视频 Latent Tokens。论文未给出 Fast-WAM 专属的统一压缩率或最大视频分辨率。
- **语言 Tokens**：Wan2.2 内置 T5 编码器将指令变成文本 Embedding；所有 Token 组通过 Cross-Attention 使用语言条件。
- **动作 Tokens**：长度 \(H=32\) 的连续动作 Chunk，作为 Flow Matching 的目标变量；论文正文未明确列出动作向量各维的物理定义。
- **输出**：机器人未来 32 步动作 Chunk；外层自回归执行／重规划循环在本文分析中被省略。

## 5. 方法步骤、计算与变量依赖（Fig.2）

**训练阶段：**
1. **观测与语言编码**：多相机图像拼接→VAE→首帧干净 Latent \(z_0\)；任务指令→T5→文本条件 \(c_l\)。二者构成共同条件。
2. **构造视频训练样本**：未来视频经 VAE 得到 \(z_{1:T}\)，采样高斯噪声 \(\epsilon_v\) 与时间 \(t_v\)，构造 \(z_t=(1-t_v)z+t_v\epsilon_v\)。Video DiT 输入 \(z_t,t_v,z_0,c_l\)，学习未来视频速度场。
3. **构造动作训练样本**：真实动作 Chunk \(a_{1:H}\) 加噪：\(a_t=(1-t_a)a+t_a\epsilon_a\)。Action Expert 输入 \(a_t,t_a\) 及由当前观测产生的世界表征、语言条件，预测动作速度场。
4. **联合优化**：两分支都拟合 Flow Matching 目标 \(\epsilon-y\)：  
   \[
   \mathcal L=\mathcal L_{\rm act}+\lambda\mathcal L_{\rm vid},\quad
   \mathcal L_{\rm FM}=\mathbb E\|f_\theta(y_t,t,o,l)-(\epsilon-y)\|_2^2.
   \]
   \(\lambda\) 控制视频辅助目标的权重。视频损失训练 Video DiT 学到运动／交互表征；动作损失训练 Action Expert 预测控制序列。
5. **信息隔离（Attention Mask）**：未来视频 Tokens 在视频分支内双向注意，并可读取首帧；动作 Tokens 在动作分支内双向注意，也可读取首帧；**动作 Tokens 不可读取未来视频 Tokens**；首帧 Tokens 不读取其他 Tokens。这样两个任务共享当前场景锚点，但动作训练不依赖推理时不存在的未来信息。

**推理阶段：**
1. 当前多相机图像拼接并经 VAE 得到首帧干净 Latent；T5 编码指令。
2. 不创建未来视频噪声 Tokens，也不运行未来视频去噪；Video DiT 对当前观测做**一次前向编码**，输出世界表征 \(z(o,l)\)。
3. Action Expert 从噪声动作 Chunk 出发，条件化于 \(z(o,l)\) 和语言 Embedding，执行 **10 步 Flow Matching 去噪**，输出 32 步动作。论文设置 CFG=1.0。

**容易混淆的点：**“单次前向”指视频骨干作为世界编码器只跑一次，不代表整个动作策略只有一次计算；动作分支仍有 10 步去噪。

## 6. 基线架构、参数与本文改动

| 项目 | 论文实现 |
|---|---|
| 视频基线／骨干 | Wan2.2-5B 预训练视频 DiT，连同其 T5 文本编码器、视频 VAE |
| 动作分支 | Action Expert DiT；架构类型与视频分支相同，隐藏维度缩至 \(d_a=1024\)，约 1B 参数 |
| 总参数 | 约 6B（5B 视频骨干＋1B 动作专家） |
| 结构 | Mixture-of-Transformers（MoT），视频与动作分支共享 Attention；T5 条件通过 Cross-Attention 注入 |
| Attention | 通过结构化 Mask 限制首帧、未来视频、动作 Tokens 的可见关系；不是让动作直接读取未来视频 |
| FFN | DiT／Action Expert 内的前馈层；论文未报告足以逐层复现的 FFN 中间维度、激活函数等完整细节 |
| 训练／推理精度 | 论文称混合精度训练；没有给出可概括为统一 FP16/BF16/FP8 的完整训推规格 |
| 训练配置 | 动作长度 32；视频时间下采样 4×，每个 chunk 9 帧；AdamW，学习率 \(10^{-4}\)，weight decay 0.01 |
| 推理配置 | 10 步去噪，CFG=1.0；延迟在单张 RTX 5090D V2 32GB 上测量 |

**与基线最本质的差异：** Fast-WAM 并未删除视频模型或视频训练损失，而是把视频 DiT 从“在线未来视频生成器”改成“训练受视频目标塑造、推理单次调用的世界编码器”。

## 7. 实验、指标与有效性

| 测试 | Fast-WAM | 关键对照 | 结论 |
|---|---:|---:|---|
| RoboTwin 2.0 平均成功率 | **91.8%** | LingBot-VA 92.2%；Motus 87.8% | 无具身预训练，接近强预训练 WAM |
| LIBERO 平均成功率 | **97.6%** | π₀.₅ 96.9%；LingBot-VA 98.5%；Motus 97.7% | 无具身预训练，表现有竞争力 |
| RoboTwin 消融 | 91.8% | Joint 90.6%；IDM 91.3%；无视频协同 83.8% | 移除视频协同的损失明显大于改变推理范式 |
| LIBERO 消融 | 97.6% | Joint 98.5%；IDM 98.0%；无视频协同 93.5% | 同样支持训练期视频目标的重要性 |
| 真实毛巾折叠延迟 | **190 ms** | Joint 580 ms；IDM 810 ms | 比两种 imagine-then-execute 变体分别约快 3.1×、4.3× |

评测设置：LIBERO 四个子集、40 个任务共 2000 次 trial；RoboTwin 2.0 为双臂任务，超过 50 个任务，包含 clean／randomized 场景；真实任务为 Galaxea R1 Lite 毛巾折叠，使用 60 小时遥操作示范。真实任务中，预训练 π₀.₅ 仍是整体最强；Fast-WAM 家族内部 IDM 成功率最高，Fast-WAM 完成时间更好。移除视频协同后真实任务成功率降至 10%。详见 Fig.3–4、Table 1–2、Appendix Table 3。

## 8. 贡献总结

1. 将 WAM 的两个因素——训练期视频预测目标、推理期未来视频生成——通过受控变体拆开分析。
2. 提出 Fast-WAM：保留视频协同训练，推理时将视频 DiT 用作单次世界表征编码器，动作分支直接生成控制序列。
3. 实验显示，去掉视频协同训练造成的性能损失，显著大于去掉推理期未来想象的损失。
4. 在无具身预训练条件下，在 LIBERO、RoboTwin 及真实操作任务上取得竞争力，并展示低延迟部署。

## 9. 局限与有针对性的研究问题

- **“学到世界模型”尚缺少表征层证据：** 当前主要以动作成功率间接证明世界表征有效。可对同一首帧构造多个候选动作，检查 latent 是否能区分对应未来状态／接触结果；并测试该表征能否迁移到训练中未出现的动作组合。
- **视频目标如何影响动作分支尚未拆清：** 论文证明 co-training 有效，但没有充分区分收益来自共享 Attention、视频分支参数更新，还是辅助损失的正则化。可做冻结 Video DiT、仅共享 Attention、停止梯度、分层共享等成组对照，并报告动作性能与表征变化。
- **“未来视频不必要”的结论受任务与动作 Chunk 限制：** 论文聚焦单个 32 步 chunk，省略外层自回归 rollout；对需要多阶段接触、遮挡后状态判断或失败恢复的任务，显式未来信息可能仍有价值。应在相同策略骨干下按任务阶段比较 direct latent policy 与 video-conditioned action 的差异。
- **真实评测覆盖面有限：** 真实机器人核心验证集中在毛巾折叠；柔性物体任务很有价值，但不足以代表刚体精细装配、工具使用、移动操作等控制结构。需要按任务机制分层验证，而不是简单扩大任务数量。

## 模型信息字段汇总

| 字段 | Fast-WAM |
|---|---|
| 生成范式 | 训练：视频＋动作双 Flow Matching；推理：动作 Flow Matching |
| 模型名称 | Fast-WAM |
| 所属公司／机构 | Galaxea AI；清华大学 IIIS |
| 时间 | 2026-03（arXiv v2：03-23） |
| 开源／闭源 | **开源代码与公开权重**（GitHub、Hugging Face） |
| 模型大小（B） | 约 6B（Wan2.2-5B＋约 1B Action Expert） |
| 上下文长度 | 非 LLM 自回归上下文；论文未给统一 Token 上下文长度 |
| Attention | MoT 双分支共享 Attention；结构化 Mask；T5 Cross-Attention |
| FFN | DiT 系列 FFN；完整逐层配置未披露 |
| MTP | 不适用／未报告 |
| 训推位宽 | 训练混合精度；具体统一位宽未披露 |
| 可用场景 | LIBERO、RoboTwin 2.0、Galaxea R1 Lite 毛巾折叠等已验证操作任务 |
| 视频压缩率 | Wan2.2 VAE Latent；Fast-WAM 论文未给出独立压缩率数值 |
| 最大视频规格 | Fast-WAM 实际最大输入规格未明确报告；不能直接把 Wan2.2 基础模型上限当作 Fast-WAM 上限 |
| Prompt 上下文长度 | 使用 Wan2.2 内置 T5；最大 Prompt Token 数未明确报告 |

**阅读结论：** Fast-WAM 的关键启示是“训练时需要预测未来”不等于“部署时必须生成未来”。它把 WAM 的在线成本从未来视频多步生成转为当前观测的一次视频骨干编码＋动作专家去噪。真正尚待回答的问题，是这种隐式世界表征在更复杂、长时域、需要失败恢复的闭环任务中，何时足以替代显式未来状态。

**来源：** [论文正文（v2）](https://arxiv.org/html/2603.16666v2) · [官方项目页](https://yuantianyuan01.github.io/FastWAM/) · [官方代码与模型说明](https://github.com/yuantianyuan01/FastWAM)


---

<a id="english"></a>

# Fast-WAM: Do World Action Models Need Test-time Future Imagination?

> **Paper:** [arXiv:2603.16666](https://arxiv.org/abs/2603.16666) (v2, 2026-03-23) | **Authors:** Tianyuan Yuan et al. | **Affiliations:** IIIS, Tsinghua University; Galaxea AI  
> **In one sentence:** The main benefit of a WAM may come from *learning to predict the future during training*, not from *generating future video before every action*. Fast-WAM retains video co-training but, at inference, encodes only the current observation and then generates actions.

[中文](#中文)　|　[English](#english)　|　[Paper](https://arxiv.org/abs/2603.16666)　|　[Code](https://github.com/yuantianyuan01/FastWAM)

## 1. Problem

Many WAMs use an **imagine-then-execute** pipeline: generate future video from the current image and task instruction, then predict robot actions from the imagined future. Video generation requires iterative denoising and increases inference latency. The paper asks whether WAM performance comes from video-prediction supervision during training or explicit future-video generation at inference. Earlier designs usually combine both, making their contributions hard to separate.

## 2. Existing paradigms

| Paradigm | Examples／mechanism | Strength | Main cost |
|---|---|---|---|
| Direct VLA policy | OpenVLA, π₀／π₀.₅: image + language → action | Direct inference; benefits from semantic knowledge in vision-language pretraining | Future dynamics are generally learned implicitly rather than through an explicit future-video objective |
| Joint-generation WAM | Unified World Models, Motus: jointly denoise future-video and action tokens | Models video and action in one generative process | Must process both video and action generation variables at inference |
| Video-then-action WAM | Vidar, Causal World Modeling, LingBot-VA-related line: generate future observations, then use IDM／action head | Actions explicitly depend on predicted future states | Video generation adds a costly stage before action execution |
| Fast-WAM | Keep video co-training during training; remove future-video branch at inference | Retains video supervision without online future-video synthesis | Does not provide the action module with explicitly generated future frames |

> The paper groups prior WAMs into two main structures (Fig. 1): **joint denoising** and **video-then-action**. The named works are representative related approaches, not necessarily reproduced under identical conditions in this paper.

## 3. Difference from baselines

Fast-WAM changes the **inference path**, rather than simply replacing the video model with a smaller one. It uses the pretrained Wan2.2-5B video DiT as a world-representation encoder. Future-video latents remain a training target; at inference, only the clean latent of the current observation is passed through the video DiT once. The resulting world representation conditions an Action Expert that generates actions.

The paper builds three controlled variants (Sec. 3.3):
- **Fast-WAM-Joint:** future-video and action tokens are jointly denoised with shared attention; action generation remains coupled to video generation.
- **Fast-WAM-IDM:** generate future-video tokens first, then condition the action module on the resulting future representation.
- **Fast-WAM without video co-training:** keep the architecture and inference unchanged, but remove the video-prediction loss to isolate the effect of video co-training.

## 4. Inputs, outputs, and representations

- **Input:** Current multi-camera RGB observations \(o\) and a natural-language task instruction \(l\). Camera images are concatenated into one image before entering the pretrained Wan2.2 VAE.
- **Visual tokens:** The VAE maps images／video into continuous latents. The current first frame is represented by clean visual tokens; future frames become future-video latent tokens. The paper does not specify a Fast-WAM-specific compression ratio or maximum video resolution.
- **Language tokens:** Wan2.2's built-in T5 encoder converts the instruction into text embeddings. All token groups receive language conditioning through Cross-Attention.
- **Action tokens:** A continuous action chunk of length \(H=32\), used as the Flow-Matching target. The paper does not explicitly list the physical meaning of every action-vector dimension in the main text.
- **Output:** A 32-step robot action chunk. The outer autoregressive execution／replanning loop is omitted from this study.

## 5. Pipeline, computations, and variable dependencies (Fig. 2)

**Training:**
1. **Encode observation and language:** concatenate camera images → VAE → clean first-frame latent \(z_0\); task instruction → T5 → language condition \(c_l\). These provide shared context.
2. **Build the video training sample:** encode future video with the VAE to obtain \(z_{1:T}\); sample Gaussian noise \(\epsilon_v\) and time \(t_v\); interpolate \(z_t=(1-t_v)z+t_v\epsilon_v\). Video DiT receives \(z_t,t_v,z_0,c_l\) and learns the future-video velocity field.
3. **Build the action training sample:** add noise to the ground-truth action chunk \(a_{1:H}\): \(a_t=(1-t_a)a+t_a\epsilon_a\). The Action Expert receives \(a_t,t_a\), the world representation from the current observation, and language conditioning; it predicts the action velocity field.
4. **Joint optimization:** both branches fit Flow-Matching targets \(\epsilon-y\):  
   \[
   \mathcal L=\mathcal L_{\rm act}+\lambda\mathcal L_{\rm vid},\quad
   \mathcal L_{\rm FM}=\mathbb E\|f_\theta(y_t,t,o,l)-(\epsilon-y)\|_2^2.
   \]
   \(\lambda\) weights the auxiliary video objective. The video loss shapes motion／interaction representations in Video DiT; the action loss trains the Action Expert to predict control sequences.
5. **Attention isolation:** future-video tokens attend bidirectionally within the video branch and can read the first-frame tokens; action tokens attend bidirectionally within the action branch and can also read the first frame; **action tokens cannot read future-video tokens**; first-frame tokens do not read other tokens. Both tasks share the current-scene anchor without making action training depend on future information unavailable at inference.

**Inference:**
1. Concatenate current camera images and encode them into clean first-frame latents with the VAE; encode the instruction with T5.
2. Do not instantiate noisy future-video tokens and do not run future-video denoising. The Video DiT performs **one forward encoding pass** over the current observation and outputs world representation \(z(o,l)\).
3. The Action Expert starts from noisy action tokens, conditions on \(z(o,l)\) and language embeddings, and performs **10 Flow-Matching denoising steps** to output a 32-step action chunk. The paper uses CFG=1.0.

**Important distinction:** “Single pass” refers to the video backbone acting as a world encoder. It does not mean the entire action policy runs only once; the action branch still uses 10 denoising steps.

## 6. Baseline architecture, parameters, and changes

| Item | Paper implementation |
|---|---|
| Video backbone | Pretrained Wan2.2-5B video DiT, including its T5 text encoder and video VAE |
| Action branch | Action Expert DiT; same architecture type as video branch, reduced hidden dimension \(d_a=1024\), approximately 1B parameters |
| Total parameters | Approximately 6B (5B video backbone + 1B action expert) |
| Structure | Mixture-of-Transformers (MoT), shared Attention between video and action branches; T5 conditioning via Cross-Attention |
| Attention | Structured masks control visibility among first-frame, future-video, and action tokens; actions do not directly read future-video tokens |
| FFN | Feed-forward layers in DiT／Action Expert; full intermediate dimensions, activations, and layer-wise details are not reported |
| Training／inference precision | Mixed-precision training is reported; no complete unified FP16／BF16／FP8 training-and-inference specification |
| Training setup | Action horizon 32; video temporally downsampled by 4×, giving 9 frames per chunk; AdamW, LR \(10^{-4}\), weight decay 0.01 |
| Inference setup | 10 denoising steps, CFG=1.0; latency measured on one RTX 5090D V2 32GB GPU |

**Fundamental difference:** Fast-WAM does not remove the video model or video training loss. It turns the video DiT from an *online future-video generator* into a *world encoder trained with video supervision and called once at inference*.

## 7. Experiments, metrics, and ablations

| Evaluation | Fast-WAM | Key comparison | Finding |
|---|---:|---:|---|
| RoboTwin 2.0 average success | **91.8%** | LingBot-VA 92.2%; Motus 87.8% | No embodied pretraining; close to strong pretrained WAM |
| LIBERO average success | **97.6%** | π₀.₅ 96.9%; LingBot-VA 98.5%; Motus 97.7% | Competitive without embodied pretraining |
| RoboTwin ablation | 91.8% | Joint 90.6%; IDM 91.3%; no video co-training 83.8% | Removing video co-training hurts more than changing inference paradigm |
| LIBERO ablation | 97.6% | Joint 98.5%; IDM 98.0%; no video co-training 93.5% | Same evidence for training-time video objective |
| Real towel-folding latency | **190 ms** | Joint 580 ms; IDM 810 ms | About 3.1× and 4.3× faster, respectively |

Evaluation setup: LIBERO has four suites and 40 tasks, evaluated over 2,000 trials; RoboTwin 2.0 includes more than 50 bimanual tasks under clean／randomized scenes; the real task is towel folding on Galaxea R1 Lite, trained with 60 hours of teleoperated demonstrations. In the real task, pretrained π₀.₅ remains the strongest overall method; among Fast-WAM variants, IDM has the highest success rate while Fast-WAM has better completion time. Removing video co-training reduces real-task success to 10%. See Fig. 3–4, Tables 1–2, and Appendix Table 3.

## 8. Contributions

1. Separates two WAM factors—training-time video prediction and inference-time future-video generation—using controlled variants.
2. Proposes Fast-WAM: retain video co-training, use Video DiT as a single-pass world encoder at inference, and generate actions directly with the action branch.
3. Shows that removing video co-training causes a much larger performance drop than removing test-time future imagination.
4. Achieves competitive results without embodied pretraining on LIBERO, RoboTwin, and a real manipulation task, with low inference latency.

## 9. Limitations and targeted research questions

- **The “world representation” claim lacks direct representation-level evidence:** Current evidence is mainly indirect, via action success. For the same first frame, provide multiple candidate actions and test whether latents distinguish their resulting future states／contact outcomes; test transfer to unseen action combinations.
- **How video objectives affect the action branch is not isolated:** Co-training is shown to help, but the paper does not fully separate gains from shared Attention, video-branch parameter updates, or auxiliary-loss regularization. Compare frozen Video DiT, Attention-only sharing, stop-gradient, and hierarchical sharing; report both action performance and representation changes.
- **“Future video is unnecessary” is bounded by task and chunk design:** The study focuses on a single 32-step chunk and omits the outer autoregressive rollout. Explicit future information may still matter for multi-stage contact, post-occlusion state estimation, or failure recovery. Compare latent direct policies with video-conditioned actions across task phases under the same backbone.
- **Real-world coverage is narrow:** The main physical validation is towel folding. Deformable-object manipulation is valuable, but does not represent rigid precision assembly, tool use, or mobile manipulation. Evaluate by task mechanism rather than merely increasing task count.

## Model information summary

| Field | Fast-WAM |
|---|---|
| Generation paradigm | Training: video + action Flow Matching; inference: action Flow Matching |
| Model name | Fast-WAM |
| Organization／company | Galaxea AI; IIIS, Tsinghua University |
| Date | March 2026 (arXiv v2: Mar 23) |
| Open／closed source | **Open-source code and released checkpoints** (GitHub, Hugging Face) |
| Model size (B) | Approx. 6B (Wan2.2-5B + approx. 1B Action Expert) |
| Context length | Not an LLM autoregressive context; no unified token context length reported |
| Attention | Two-branch MoT with shared Attention; structured masks; T5 Cross-Attention |
| FFN | DiT-family FFN; full layer-wise details not disclosed |
| MTP | Not applicable／not reported |
| Training／inference precision | Mixed-precision training; exact unified precision not disclosed |
| Use cases | LIBERO, RoboTwin 2.0, and evaluated manipulation tasks such as towel folding on Galaxea R1 Lite |
| Video compression ratio | Wan2.2 VAE latents; no independent Fast-WAM compression ratio reported |
| Maximum video specification | Fast-WAM's actual maximum input specification is not clearly reported; do not equate it with the Wan2.2 base-model limit |
| Prompt context length | Uses Wan2.2's built-in T5; maximum prompt-token count not clearly reported |

**Takeaway:** “Predict the future during training” does not imply “generate the future during deployment.” Fast-WAM shifts online cost from multi-step future-video generation to one current-observation encoding pass plus action-expert denoising. The open question is when this implicit world representation can replace explicit future states in more complex, long-horizon closed-loop tasks that require recovery from failure.

**Sources:** [Paper (v2)](https://arxiv.org/html/2603.16666v2) · [Project page](https://yuantianyuan01.github.io/FastWAM/) · [Official code and model notes](https://github.com/yuantianyuan01/FastWAM)
