'''
Date: 2023-06-25 07:15:33
LastEditors: Lionel 1334252492@qq.com
LastEditTime: 2023-09-04 04:21:55
FilePath: /KRDN_Speed/utils/parser.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="RoSA-KG")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="ml-1m2", help="Choose a dataset:[last-fm,alibaba-ifashion,yelp2018,mind-f,amazon-book,MIND]")
    parser.add_argument("--data_path", nargs="?", default="data/", help="Input data path.")
 
    # ===== train ===== #
    parser.add_argument('--epoch', type=int, default=300, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=1024, help='test batch size')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate') #0.0005-->yelp 2018  
    parser.add_argument('--gamma', type=float, default=0.5, help='drop threshold')
    parser.add_argument('--lr_dc_step', type=float, default=100, help='drop threshold')
    parser.add_argument('--lr_dc', type=float, default=0.1, help='drop threshold')
    parser.add_argument('--max_iter', type=float, default=2, help='iteration times')
    parser.add_argument("--inverse_r", type=bool, default=False, help="consider inverse relation or not")
    parser.add_argument("--node_dropout", type=bool, default=True, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=1, help="ratio of node dropout") #yelp-->0.75 last-fm-->0.3
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--channel", type=int, default=100 , help="hidden channels for model")
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id; set -1 for auto-pick by free memory")
    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]', help='Output sizes of every layer')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument('--kg_edge_sample_rate', type=float, default=0.2,
                        help='Sampling rate for knowledge graph edges during training (0, 1].')
    parser.add_argument('--ui_edge_sample_rate', type=float, default=0.2,
                        help='Sampling rate for user-item edges during training (0, 1].')
    parser.add_argument('--kg_edge_chunk_size', type=int, default=1024,
                        help='Chunk size for iterating knowledge graph edges.')
    parser.add_argument('--ui_edge_chunk_size', type=int, default=1024,
                        help='Chunk size for iterating user-item edges.')
    parser.add_argument('--dynamic_edge_chunk_size', type=int, default=20480,
                        help='Chunk size for batching dynamic edge injection to reduce memory pressure.')
    parser.add_argument('--mmd_estimator', nargs='?', default='linear', choices=['linear', 'full'],
                        help='Estimator for MMD loss (linear avoids quadratic kernels).')
    parser.add_argument('--mmd_bandwidth_sample', type=int, default=1024,
                        help='Sample size used to estimate Gaussian kernel bandwidth for MMD.')
    parser.add_argument('--mmd_batch_size', type=int, default=1024,
                        help='Triplet sample size for MMD computation.')
    parser.add_argument('--kg_neg_sample_rate', type=float, default=1.0,
                        help='Sampling rate for KG negative sampling pool.')
    parser.add_argument('--neg_pool_workers', type=int, default=0,
                        help='Max workers for negative-sampling pool (0 uses auto adaptive value).')
    parser.add_argument('--neg_pool_use_threads', action='store_true',
                        help='Force thread-based pool for negative sampling to reduce process spawning.')
 
    # ===== relation context ===== #
    parser.add_argument('--context_hops', type=int, default=2, help='number of context hops')

    parser.add_argument('--num_neg_sample', type=int, default=1, help='the number of negative sample')
    parser.add_argument('--margin', type=float, default=0.2, help='the margin of contrastive_loss')
    parser.add_argument('--loss_f', nargs="?", default="contrastive_loss",
                        help="Choose a loss function:[inner_bpr, contrastive_loss]")

    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=False, help="save model or not")
    parser.add_argument("--out_dir", type=str, default="./model_para/", help="output directory for model")

    # ===== dynamic rules ===== #
    # NOTE: We mine rules statically once, then do dynamic structural role training
    parser.add_argument("--enable_role_training", type=bool, default=True,
                        help="Enable structural role dynamic training (rules are mined once statically).")
    # Backward compatibility; deprecated alias of enable_role_training
    parser.add_argument("--enable_dynamic_rules", type=bool, default=None,
                        help="[Deprecated] Alias of --enable_role_training; rules are mined once, roles are dynamic.")
    parser.add_argument("--rule_max_length", type=int, default=3,
                        help="Maximum length of rule bodies to enumerate.")
    parser.add_argument("--rule_min_support", type=int, default=10,
                        help="Minimum support threshold for candidate rules.")
    parser.add_argument("--rule_min_confidence", type=float, default=0.2,
                        help="Minimum confidence threshold for candidate rules.")
    parser.add_argument("--rule_min_pca_confidence", type=float, default=0.3,
                        help="Minimum PCA confidence threshold for candidate rules.")
    parser.add_argument("--rule_topk_per_length", type=int, default=500,
                        help="Keep top-k rules per body length after mining.")
    parser.add_argument("--rule_max_body_evaluations", type=int, default=50000,
                        help="Max number of rule bodies to evaluate across all lengths (controls runtime).")
    parser.add_argument("--rule_max_rules", type=int, default=2000,
                        help="Maximum number of rules kept after filtering.")
    parser.add_argument("--rule_min_lift", type=float, default=1.0,
                        help="Minimum lift value required during post-processing.")
    parser.add_argument("--rule_min_conviction", type=float, default=1.0,
                        help="Minimum conviction value required during post-processing.")
    parser.add_argument("--rule_refresh_interval", type=int, default=3,
                        help="Epoch interval for refreshing dynamic rules.")
    parser.add_argument("--rule_max_roles", type=int, default=128,
                        help="Maximum number of active roles per refresh cycle.")
    parser.add_argument("--rule_edges_per_role", type=int, default=512,
                        help="Maximum number of virtual edges injected per role.")
    parser.add_argument("--rule_force_rebuild", type=bool, default=False,
                        help="Force rebuilding the rule cache even if a cached file exists.")
    parser.add_argument("--rule_reward_weights", nargs="?", default="recall:1.0,ndcg:0.4,kgat_ad:0.3,kgat_arp:-0.25",
                        help="Comma separated list of metric:weight pairs for role feedback.")
    # Rule mining performance/accuracy knobs
    parser.add_argument("--rule_sample_bodies", type=bool, default=False,
                        help="Sample length>=3 rule bodies probabilistically to speed up mining (may miss low-score rules).")
    parser.add_argument("--rule_body_sample_rate", type=float, default=0.3,
                        help="Base sampling rate for body enumeration (applied to length>=3, exponent by (L-2)).")
    parser.add_argument("--rule_gpu_mining", type=bool, default=True,
                        help="Attempt GPU-accelerated rule body materialization on small graphs.")
    parser.add_argument("--rule_gpu_entity_threshold", type=int, default=400000,
                        help="Max entities to allow GPU materialization; larger graphs fall back to CPU.")
    parser.add_argument("--rule_use_torch_sparse", type=bool, default=True,
                        help="Prefer torch-sparse (SparseTensor) on CUDA for rule materialization if available.")

    # ===== structural role as entity (optional) ===== #
    parser.add_argument("--role_entity_mode", type=bool, default=True,
                        help="Inject structural roles as entity nodes (two-hop edges) instead of new relations.")
    parser.add_argument("--progress_verbose", type=bool, default=True,
                        help="Print phase progress with ETA via tqdm and stage messages.")

    # ===== curriculum for role injection (schedule over epochs) ===== #
    parser.add_argument('--role_warmup_epochs', type=int, default=2,
                        help='Warmup epochs before roles contribute (0..warmup-1).')
    parser.add_argument('--role_adjust_period', type=int, default=3,
                        help='Adjust role parameters every N epochs after warmup.')
    parser.add_argument('--role_mix_coeff_init', type=float, default=0.45,
                        help='Initial mixing coefficient for new-edge branch after warmup.')
    parser.add_argument('--role_mix_coeff_final', type=float, default=1.2,
                        help='Final mixing coefficient for new-edge branch.')
    parser.add_argument('--new_edge_threshold_init', type=float, default=0.55,
                        help='Initial hard threshold for new-edge gating after warmup (higher is stricter).')
    parser.add_argument('--new_edge_threshold_final', type=float, default=0.25,
                        help='Final hard threshold for new-edge gating (lower allows more edges).')
    parser.add_argument('--mmd_weight', type=float, default=0.35,
                        help='Weight applied to MMD regularizer aligning new-edge and KGC scores.')
    parser.add_argument('--role_relation_reg', type=float, default=1e-2,
                        help='L2 regularization weight keeping rule-specific relations close to base templates.')
    parser.add_argument('--role_injection_start_epoch', type=int, default=2,
                        help='Epoch index to start structural role injection.')
    parser.add_argument('--role_first_fraction', type=float, default=0.2,
                        help='Initial fraction of candidate roles to activate once injection starts.')
    parser.add_argument('--role_fraction_step', type=float, default=0.12,
                        help='Increment added to the active role fraction every adjustment period.')
    parser.add_argument('--role_entity_fraction', type=float, default=0.4,
                        help='Initial fraction of filtered entities retained for each active role.')
    parser.add_argument('--role_entity_fraction_step', type=float, default=0.18,
                        help='Increment added to the per-role entity fraction every adjustment period.')
    parser.add_argument('--role_npmi_threshold', type=float, default=0.25,
                        help='Minimum NPMI with anchor items to keep an entity in a role entity set.')
    parser.add_argument('--role_cos_threshold', type=float, default=0.45,
                        help='Fallback cosine similarity threshold when NPMI is insufficient.')
    parser.add_argument('--role_filter_min_keep', type=int, default=2,
                        help='Fallback minimum entities kept per role after filtering (avoid empty sets).')
    parser.add_argument('--role_topk_ratio', type=float, default=0.5,
                        help='Top-K ratio applied to filtered role entities (0~1).')
    parser.add_argument('--role_context_cap', type=int, default=64,
                        help='Maximum number of non-item context nodes kept per role after filtering.')
    parser.add_argument('--rule_reset_feedback_each_refresh', action='store_true',
                        help='Force dynamic rule trainer to clear feedback state at every refresh (legacy behaviour).')
    parser.add_argument('--role_negative_fraction_scale', type=float, default=0.8,
                        help='Scale applied to role_fraction when reward is negative.')
    parser.add_argument('--role_negative_mix_scale', type=float, default=0.88,
                        help='Scale applied to role_mix_coeff when reward is negative.')
    parser.add_argument('--role_negative_threshold_shift', type=float, default=0.01,
                        help='Increase of new_edge_threshold when reward is negative.')
    parser.add_argument('--role_positive_fraction_scale', type=float, default=1.18,
                        help='Scale applied to role_fraction when reward is positive.')
    parser.add_argument('--role_positive_mix_scale', type=float, default=1.18,
                        help='Scale applied to role_mix_coeff when reward is positive.')
    parser.add_argument('--role_positive_threshold_shift', type=float, default=-0.02,
                        help='Decrease of new_edge_threshold when reward is positive.')
    parser.add_argument('--role_ft_steps', type=int, default=5,
                        help='Fine-tune steps for newly injected role edges per chunk (0 means skip).')
    parser.add_argument('--role_ft_batch', type=int, default=256,
                        help='Batch size for role edge fine-tuning.')
    parser.add_argument('--role_pretrain_steps', type=int, default=2,
                        help='Extra KGC fine-tune steps applied to new role edges before injection.')
    parser.add_argument('--role_mix_smooth', type=float, default=0.15,
                        help='Exponential smoothing factor for role mix coefficient updates (0 disables smoothing).')
    parser.add_argument('--role_fraction_smooth', type=float, default=0.15,
                        help='Exponential smoothing applied to role/entity selection fractions.')
    parser.add_argument('--role_item_mix_momentum', type=float, default=0.55,
                        help='Momentum coefficient for per-item role mix tracking (0~1).')
    parser.add_argument('--role_min_active', type=int, default=1,
                        help='Minimum number of structural roles to keep active per refresh interval.')
    parser.add_argument('--role_round_robin_fraction', type=float, default=1.0,
                        help='Fraction of role edges preserved during round-robin sampling (0~1].')
    parser.add_argument('--role_edge_score_threshold', type=float, default=0.3,
                        help='角色边注入前的 KGC 得分阈值。')
    parser.add_argument('--role_edge_top_ratio', type=float, default=0.6,
                        help='角色边得分排序后保留的 Top-K 比例 (0~1]。')
    parser.add_argument('--role_edge_min_keep', type=int, default=2,
                        help='角色边过滤后至少保留的边数量。')
    parser.add_argument('--kgr_epoch_initial', type=int, default=9,
                        help='Number of KGR fine-tune epochs during the very first injection pass.')
    parser.add_argument('--kgr_epoch_refresh', type=int, default=3,
                        help='Number of KGR fine-tune epochs for subsequent refresh passes.')
    parser.add_argument('--role_neg_per_pos', type=int, default=4,
                        help='Number of negative samples generated per positive role edge during fine-tuning.')
    parser.add_argument('--role_neg_corrupt_ratio', type=float, default=0.5,
                        help='Fraction of negatives generated by corrupting head entities (rest corrupt tails).')
    parser.add_argument('--role_fraction_learned_weight', type=float, default=0.4,
                        help='Blend factor for learned logistic role fraction (0 disables).')
    parser.add_argument('--role_fraction_lr', type=float, default=0.35,
                        help='Learning rate applied to role selection fraction logits.')
    parser.add_argument('--role_entity_lr', type=float, default=0.25,
                        help='Learning rate applied to role entity fraction logits.')
    parser.add_argument('--role_gate_degree_coef', type=float, default=0.15,
                        help='Coefficient controlling degree-based damping for role edges.')
    parser.add_argument('--role_gate_reward_coef', type=float, default=0.2,
                        help='Coefficient to convert diversity reward into gate scaling.')
    parser.add_argument('--role_explore_epochs', type=int, default=2,
                        help='Number of consecutive epochs to keep exploration boost after detecting diversity drop.')
    parser.add_argument('--role_explore_fraction', type=float, default=0.15,
                        help='Minimum selection fraction enforced during exploration epochs.')
    parser.add_argument('--role_alignment_scale', type=float, default=0.8,
                        help='Multiplicative scale applied to role alignment penalty weight (0~1].')
    parser.add_argument('--role_memory_momentum', type=float, default=0.7,
                        help='EMA factor for role entity memory vectors (0 uses current embedding, 1 keeps previous).')
    parser.add_argument('--role_inject_score_threshold', type=float, default=0.1,
                        help='Minimum predicted KGC score for role edges to pass injection gate.')
    parser.add_argument('--role_inject_phase_ratio', type=float, default=0.6,
                        help='Fraction of filtered role edges actually injected per refresh (0~1].')
    parser.add_argument('--role_selection_floor', type=float, default=0.0,
                        help='Lower bound on role selection/entity fractions after penalties.')
    parser.add_argument('--role_selection_cap', type=float, default=0.6,
                        help='Upper bound on role selection/entity fractions (0~1].')
    parser.add_argument('--role_negative_backoff', type=float, default=0.4,
                        help='Multiplicative backoff applied to role fractions when reward drops sharply.')
    parser.add_argument('--role_metric_drop_threshold', type=float, default=0.002,
                        help='Absolute reward drop needed to trigger automatic backoff.')
    parser.add_argument('--role_reward_floor', type=float, default=0.0,
                        help='Reward floor; negative rewards beyond this trigger a temporary injection skip.')
    parser.add_argument('--role_prescore_enable', type=bool, default=True,
                        help='Enable pre-injection scoring to simulate role impact before writing to graph.')
    parser.add_argument('--role_prescore_alpha', type=float, default=0.6,
                        help='Weight of KGC-based component in role pre-score (0~1).')
    parser.add_argument('--role_prescore_beta', type=float, default=0.3,
                        help='Weight of embedding cosine component in role pre-score (0~1).')
    parser.add_argument('--role_prescore_gamma', type=float, default=0.15,
                        help='Weight of reward component in role pre-score (0~1).')
    parser.add_argument('--role_prescore_blend', type=float, default=0.45,
                        help='Blend ratio between raw candidate score and pre-score when ranking roles.')
    parser.add_argument('--role_prescore_skip', type=float, default=0.25,
                        help='If pre-score falls below this value, defer injection for the role.')
    parser.add_argument('--role_prescore_gate_floor', type=float, default=0.35,
                        help='Minimum scaling factor applied to gate weights when pre-score is low.')
    parser.add_argument('--role_prescore_role_cap', type=float, default=0.35,
                        help='Maximum fraction of roles allowed when global pre-score is low.')
    parser.add_argument('--role_log_file', type=str, default='',
                        help='File path used to persist role injection diagnostics (default result/<dataset>_role_log.txt).')

    return parser.parse_args()
