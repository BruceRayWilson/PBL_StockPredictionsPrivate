# filename: stock_manager.py

import argparse
from datetime import datetime
import pandas as pd
import sys

import os
import glob


from StockSymbolCollection.StockSymbolCollection import StockSymbolCollection
from StockData.StockData import StockData
from StockPreprocessor.StockPreprocessor import StockPreprocessor
from StockDNA.StockDNA import StockDNA
from TrainPreparation.TrainPreparation import TrainPreparation 
from LLM.LLM import LLM



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







import argparse

def add_args():
    """Function to add command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--menu", action='store_true', help="Enable the menu (default: %(default)s)")
    parser.add_argument("-ssv", "--stock_symbol_verification", action="store_true",
                        help="Flag to create a stock collection (default: %(default)s)")
    parser.add_argument("-tb", "--train_base_filename", default='train_base.csv',
                        help="CSV filename for StockSymbolCollection (default: %(default)s)")
    parser.add_argument("-tf", "--train_filename", default='train.csv',
                        help="CSV filename to get the list of stock symbols for StockData (default: %(default)s)")
    parser.add_argument("-csd", "--collect_stock_data", action='store_true',
                        help="Flag to collect stock data (default: %(default)s)")
    parser.add_argument("-st", "--start_time", default="2010-01-01",
                        help="Start time for StockData in the format YYYY-MM-DD (default: %(default)s)")
    parser.add_argument("-et", "--end_time", default="2023-10-01",
                        help="End time for StockData in the format YYYY-MM-DD (default: %(default)s)")
    parser.add_argument("-pp", "--preprocess", action='store_true', help="Execute StockPreprocessor (default: %(default)s)")
    parser.add_argument("-sdna", "--stock_dna", action='store_true', help="Execute Stock DNA (default: %(default)s)")
    parser.add_argument("-tp", "--train_preparation", action='store_true', help="Execute Train Preparation (default: %(default)s)")
    parser.add_argument("-train", action='store_true', help="Execute LLM training (default: %(default)s)")
    parser.add_argument("-predict", action='store_true', help="Execute LLM prediction (default: %(default)s)")

    # Parse the command-line arguments
    args = parser.parse_args()
    return args


def main() -> None:
    """Main function to execute the script"""
    args = add_args()
    if args.menu:
        while True:
            menu(args)
    else:
        if args.stock_symbol_verification:
            StockSymbolCollection.exec(args.train_base_filename)
        if args.collect_stock_data:
            start_time = datetime.strptime(args.start_time, '%Y-%m-%d')
            end_time   = datetime.strptime(args.end_time,   '%Y-%m-%d')
            StockData.exec(args.train_filename, start_time, end_time)
        if args.preprocess:
            StockPreprocessor.exec()
        if args.stock_dna:
            StockDNA.exec(StockPreprocessor.chunk_size)
        if args.train_preparation:
            TrainPreparation.exec()
        if args.train:
            LLM.train()
        if args.predict:
            LLM.predict()

    print("Done!")


def menu(args) -> None:
    """
    Displays a menu for users who run the script without CLI arguments
    """
    while True:
        print("1. Stock Symbol Verification")
        print("2. Stock Data Collection")
        print("3. Stock Preprocessor")
        print("4. Stock DNA")
        print("5. Train Preparation")
        print("6. LLM")
        print("   6.1 LLM Training")
        print("   6.2 LLM Prediction")
        print("Q. Quit")

        choice = input("Choose an option (1-6, Q to quit): ")

        if choice.upper() == 'Q':
            print("Exiting the program.")
            sys.exit(0)
        elif choice == '1':
            csv_filename = 'train_base.csv'
            StockSymbolCollection.exec(csv_filename)
        elif choice == '2':
            start_time = datetime.strptime(args.start_time, '%Y-%m-%d')
            end_time   = datetime.strptime(args.end_time,   '%Y-%m-%d')
            StockData.exec(args.train_filename, start_time, end_time)
        elif choice == '3':
            StockPreprocessor.exec()
        elif choice == '4':
            StockDNA.exec(StockPreprocessor.chunk_size)
        elif choice == '5':
            TrainPreparation.exec()
        elif choice == '6':
            subchoice = input("   Choose an option (1 or 2): ")
            if subchoice == '1':
                LLM.train()
            elif subchoice == '2':
                LLM.predict_string()


if __name__ == "__main__":
    main()

"""
This script creates the necessary classes and methods. You can run the script from the 
command line using CLI arguments (Option A), or run the script and use the menu (Option B). 

Please note the current state of the methods `StockSymbolCollection.exec, StockData.exec(), 
StockPreprocessor.exec(), LLM.train() and LLM.predict()` are simple, meaning they just print 
out some information and do not perform actual operations. You would need to replace the 
print statements with your actual implementation.

You can execute option A by providing the proper arguments in the command line after the python command:

    python stock_manager.py --start_time 2023-01-01 --end_time 2023-10-01

For Option B, the menu gets displayed and expects a user input:

    python stock_manager.py
    1. Stock Symbol Verification
    2. Stock Data Collection
    3. Stock Preprocessor
    4. LLM
       4.1 LLM Training
       4.2 LLM Prediction
    Choose an option: '1'
"""

