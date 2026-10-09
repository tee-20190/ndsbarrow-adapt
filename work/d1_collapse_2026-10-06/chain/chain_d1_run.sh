# D1-COLLAPSE-1 production chain for one sequential session (PACKET "Sequence"). Sourced by HR1's frozen launcher
# work/holdout_region_2026-10/step2/run_chain.sh on the Colab VM. Order fixed (d1_gates.py MANDATORY); each do_*
# checks its own prerequisites (chain_lib_d1.sh need). Staging of the Lower Saxony FIT windows and the Verden DEV
# windows precedes any training; step 0 and parity check the environment; the all-local plan of every seed of the
# session is checked on the staged table; the session's runs train in order (each final weight announced
# WEIGHTS_READY for the driver's harvest); one DEV pass of the runs marked for inference follows.
source "$CHAIN_DIR/chain_lib_d1.sh"
do_lint
do_env
do_pins
do_roles
do_stage_ls
do_stage_dev
do_void
do_step0
do_parity
do_plan_check
do_train
do_dev
do_complete
