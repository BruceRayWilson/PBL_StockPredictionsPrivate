cd train_data

head -n 1 $(ls *.csv | head -n 1) > ../master_train_data/master.csv
head -n 1 $(ls *.csv | head -n 1) > ../test_data/test.csv

for file in *.csv; do
    tail -n +2 "$file" >> ../master_train_data/master.csv
done

cd ..
ll master_train_data

mv master_train_data/* .
head -n 100000 master.csv > master_train_data/master.csv
tail -n 50000 master.csv > test_data/test.csv
