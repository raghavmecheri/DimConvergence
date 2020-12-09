rm -rf data/
mkdir data/
rm -rf *.out
nohup python -u exp1.py &> exponelog.out &
nohup python -u exp2.py &> exptwolog.out &
