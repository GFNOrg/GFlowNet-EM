# Grammar Induction
### Edward Hu, Moksh Jain, Nikolay Malkin

This directory contains the source code to reproduce the grammar induction experiment in our paper

**GFlowNet-EM for Learning Compositional Latent Variable Models** <br>
*Edward J. Hu\*, Nikolay Malkin\*, Moksh Jain, Katie Everett, Alexandros Graikos, Yoshua Bengio* <br>
Paper: https://arxiv.org/abs/23XX.XXXXX <br>

We provide commands for learning either a context-free grammar or a non-context-free grammar where rule expansions depend on the parent of the LHS of the rule.
You can optionally specify an enegy-based model (EBM) as a prior to bias the learned grammar.
We provide one such EBM trained on tree shapes.

For context-free grammars, you can also reproduce the _Marginalization_ and _Exact Sampling_ baselines using the inside-outside algorithm.

## Commands to reproduce GFlowNet-EM results

For context-free grammars
```
python -Ou train.py --cuda --temperature 1.1 --data ./data/ptb_cpcfg/ --d_model 512 --nlayers 6 --tie_tgt_embedding --lr_flow 0.003 --lr_encoder 0.0001 --lr_forward 0.0001 --lr_backward 0.0001 --lr_grammar 0.001 --use_spans_f1 --grammar_update_tb_threshold_max 6 --grammar_update_tb_threshold_min 3.0 --grammar_update_tb_threshold_horizon 10000 --mcmc_steps 10 --use_off_policy_mcmc --agg_type skipmlp --grammar_type cfg
```

For context-free grammars with a tree-shape EBM
```
python -Ou train.py --cuda --temperature 1.1 --data ./data/ptb_cpcfg/ --d_model 512 --nlayers 6 --tie_tgt_embedding --lr_flow 0.003 --lr_encoder 0.0001 --lr_forward 0.0001 --lr_backward 0.0001 --lr_grammar 0.001 --use_spans_f1 --grammar_update_tb_threshold_max 4 --grammar_update_tb_threshold_min 2.0 --grammar_update_tb_threshold_horizon 10000 --mcmc_steps 10   --use_off_policy_mcmc --agg_type skipmlp --ebm_reward ./ebm_ckpts/ebm_checkpoint_last.pt --ebm_reward_temp_start 1 --ebm_reward_temp_end 1000 --ebm_reward_temp_horizon 10000 --ebm_reward_temp_schedule_type exp --grammar_type cfg
```

For non-context-free grammars
```
python -Ou train.py --cuda --temperature 1.1 --data ./data/ptb_cpcfg/ --d_model 512 --nlayers 6 --tie_tgt_embedding --lr_flow 0.003 --lr_encoder 0.0001 --lr_forward 0.0001 --lr_backward 0.0001 --lr_grammar 0.001 --use_spans_f1 --grammar_update_tb_threshold_max 7 --grammar_update_tb_threshold_min 3.5 --grammar_update_tb_threshold_horizon 10000 --mcmc_steps 10 --use_off_policy_mcmc --agg_type skipmlp --grammar_type ncfg
```

## Other baselines

_Marginalization_ for context-free grammars
```
python -Ou train.py --cuda --tjb_forward 0 --go_back_and_forward 0 --train_gfn 0 --data ./data/ptb_cpcfg/ --tie_tgt_embedding --lr_grammar 0.001 --parser_type marginalization --use_spans_f1 --grammar_type cfg
```

_Exact Sampling_ for context-free grammars
```
python -Ou train.py --cuda --tjb_forward 0 --go_back_and_forward 0 --train_gfn 0 --data ./data/ptb_cpcfg/ --tie_tgt_embedding --lr_grammar 0.001 --parser_type sample_from_posterior --use_spans_f1 --grammar_type cfg
```

_MC-EM_ for context-free grammars
```
python -Ou train.py --cuda --temperature 1000 --go_back_and_forward 0 --mc_em --train_gfn 0 --data ./data/ptb_cpcfg/ --tie_tgt_embedding --seqlen 20 --lr_grammar 0.001 --sleep_mle 0 --use_spans_f1 --mcmc_steps 1000 --grammar_type cfg
```

_MC-EM_ for non-context-free grammars
```
python -Ou train.py --cuda --temperature 1000 --go_back_and_forward 0 --mc_em --train_gfn 0 --data ./data/ptb_cpcfg/ --tie_tgt_embedding --seqlen 20 --lr_grammar 0.001 --sleep_mle 0 --use_spans_f1 --mcmc_steps 1000 --grammar_type ncfg
```

## Citation
```
@misc{hu2023gflownetem,
    title={GFlowNet-EM for Learning Compositional Latent Variable Models},
    author={Hu, Edward and Malkin, Nikolay and Jain, Moksh and Everett, Katie and Graikos, Alexandros and Bengio, Yoshua},
    year={2023},
    eprint={23XX.XXXXX},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
