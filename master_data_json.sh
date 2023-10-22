cd train_data

# head -n 1 $(ls *.json | head -n 1) > ../master_train_data/master.json

for file in *.json; do
    cat "$file" >> ../master_train_data/master.json
done

cd ..
ll master_train_data

mv master_train_data/* .
head -n 100000 master.json > master_train_data/master.json
tail -n 50000 master.json > test_data/test.json
