import os
import glob

def master_data():
    os.chdir("train_data")

    # Loop through the alphabet from Q to Z
    for letter in 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z':
        pattern = f"{letter}*.csv"
        files = glob.glob(pattern)

        if files:
            print(f"Deleting CSV files starting with the letter '{letter}':")
            for file in files:
                os.remove(file)

    csv_files = glob.glob("*.csv")
    if csv_files:
        # Copy header from the first CSV file to master.csv
        with open(csv_files[0], 'r') as first_csv:
            header = first_csv.readline()
            with open("../master_train_data/master.csv", 'w') as master:
                master.write(header)

        # Append the content (excluding headers) of each CSV file to master.csv
        with open("../master_train_data/master.csv", 'a') as master:
            for file in csv_files:
                with open(file, 'r') as csv_file:
                    csv_file.readline()  # Skip header
                    for line in csv_file:
                        master.write(line)

    os.chdir("..")

if __name__ == "__main__":
    master_data()
