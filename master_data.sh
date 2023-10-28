cd train_data

# Loop through the alphabet, starting from the specified letter
for letter in {Q..Z}; do
    # Construct the pattern to match CSV files
    pattern="${letter}*.csv"

    # Check if any files match the pattern
    files=( $pattern )

    # If files match the pattern, delete them
    if [ ${#files[@]} -gt 0 ]; then
        echo "Deleting CSV files starting with the letter '$letter':"
        rm -f $pattern
    fi
done



head -n 1 $(ls *.csv | head -n 1) > ../master_train_data/master.csv
head -n 1 $(ls *.csv | head -n 1) > ../test_data/test.csv

for file in *.csv; do
    tail -n +2 "$file" >> ../master_train_data/master.csv
done

cd ..
