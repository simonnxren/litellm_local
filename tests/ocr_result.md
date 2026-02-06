Table 5. Reward decomposition on R2R-CE Val-Unseen. denotes inclusion of the reward. The first row is the SFT-only baseline, while subsequent rows apply GRPO with different subsets of movement ($R_m$), action ($R_{\text{action}}$), and format ($R_{\text{format}}$) rewards. Removing any term degrades SR and SPL, and using all three yields the best navigation success and path efficiency.

| $R_m$ | $R_{\text{action}}$ | $R_{\text{format}}$ | SR↑ | SPL↑ |
| :--- | :--- | :--- | :--- | :--- |
| ✗ | ✗ | ✗ | 58.0 | 53.2 |
| ✓ | ✗ | ✗ | 60.7 | 55.5 |
| ✗ | ✓ | ✗ | 61.9 | 56.8 |
| ✗ | ✓ | ✓ | 59.6 | 54.7 |
| ✓ | ✓ | ✗ | 64.5 | 60.2 |
| ✓ | ✓ | ✗ | 63.4 | 59.1 |
| ✗ | ✓ | ✓ | 65.2 | 61.0 |
| ✓ | ✓ | ✓ | 68.3 | 65.2 |

reward $R_{\text{action}}$, and the format reward $R_{\text{format}}$. Table 5 reports the performance on the R2R-CE Val-Unseen split. Removing any rewards causes a noticeable degradation in navigation success and trajectory efficiency. Without $R_m$, the model exhibits unstable locomotion and oscillatory motion. Excluding $R_{\text{action}}$ weakens high-level decision consistency, while dropping $R_{\text{format}}$ leads to malformed or unstructured outputs that cannot be reliably parsed into executable commands. Combining all three rewards yields the highest success rate and SPL, validating that they provide complementary optimization signals for reasoning-aligned and control-stable policy learning.

Additional ablation. We further conduct incremental modality encoder ablations to analyze the contribution of each perception stream in the Supp. Mat. section 7.

6. Conclusion

In this work, we introduce MobileVLA-R1, a unified vision-language-action framework that bridges high-level reasoning and low-level control for quadruped robots. By decoupling structured chain-of-thought reasoning from continuous motor execution, our model achieves interpretable decision-making and robust control across diverse environments. The two-stage training paradigm, which combines supervised CoT alignment with GRPO-based reinforcement learning, effectively enhances reasoning consistency, control stability, and long-horizon execution. Extensive experiments on the VLN-CE and QUARD benchmarks, as well as real-world deployments on the Unitee Go2 robot, demonstrate the superior performance and adaptability of MobileVLA-R1 compared to existing methods. These results highlight the effectiveness of integrating structured reasoning with continuous control, advancing the development of generalizable embodied agents.
