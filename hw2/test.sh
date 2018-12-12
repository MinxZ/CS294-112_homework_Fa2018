python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -dna --exp_name sb_no_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg -dna --exp_name sb_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg --exp_name sb_rtg_na
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name sb_no_rtg_na

python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -dna --exp_name lb_no_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg -dna --exp_name lb_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg --exp_name lb_rtg_na


python plot.py data/sb_*
python plot.py data/lb_*

python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --discount 0.99 -dna --exp_name dis_sb_no_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --discount 0.99 --exp_name dis_sb_no_rtg_na
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --discount 0.99 -rtg -dna --exp_name dis_sb_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --discount 0.99 -rtg --nn_baseline --exp_name dis__base_sb_rtg_na

python plot.py data/dis_*
