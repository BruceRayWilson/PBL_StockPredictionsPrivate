import argparse
from datetime import datetime
import pandas as pd
import sys
import shutil

import os
import glob
import csv

import argparse

from config_loader import load_config


from StockSymbolCollection.StockSymbolCollection import StockSymbolCollection
from StockData.StockData import StockData
from StockPreprocessor.StockPreprocessor import StockPreprocessor
from StockDNA.StockDNA import StockDNA
from TrainPreparation.TrainPreparation import TrainPreparation 
from LLM.LLM import LLM



def master_data():
    print("Starting to create the master.csv file...")

    os.chdir("train_data")  # Move to the 'train_data' directory

    # Check if 'master_train_data' directory exists relative to 'train_data'; if not, create it
    if not os.path.exists("../master_train_data"):
        os.makedirs("../master_train_data")

    # If 'master.csv' exists in 'master_train_data', delete it
    master_csv_path = "../master_train_data/master.csv"
    if os.path.exists(master_csv_path):
        os.remove(master_csv_path)

    csv_files = glob.glob("*.csv")
    if csv_files:
        # Process each CSV file
        with open(master_csv_path, 'w', newline='') as master:
            master_writer = csv.writer(master)
            # Write header only once
            master_writer.writerow(['Sentence', 'Tomorrow_Gain', 'Gain'])
            
            for file in csv_files:
                with open(file, 'r') as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    for row in csv_reader:
                        # Extract only the required columns
                        master_writer.writerow([row['Sentence'], row['Tomorrow_Gain'], row['Gain']])

    os.chdir("..")

    print("Finished creating the master.csv file.")








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
    parser.add_argument("-et", "--end_time", default="2023-12-01",
                        help="End time for StockData in the format YYYY-MM-DD (default: %(default)s)")
    # Add new arguments for prediction start and end times with the last month's dates as defaults
    parser.add_argument("--predict_start_time", default="2023-11-01",
                        help="Start time for prediction in the format YYYY-MM-DD (default: last month's start date)")
    parser.add_argument("--predict_end_time", default="2023-12-01",
                        help="End time for prediction in the format YYYY-MM-DD (default: last month's end date)")
    parser.add_argument("-pp", "--preprocess", action='store_true', help="Execute StockPreprocessor (default: %(default)s)")
    parser.add_argument("-sdna", "--stock_dna", action='store_true', help="Execute Stock DNA (default: %(default)s)")
    parser.add_argument("-tp", "--train_preparation", action='store_true', help="Execute Train Preparation (default: %(default)s)")
    parser.add_argument("-md", "--master_data", action='store_true', help="Execute Master Data (default: %(default)s)")
    parser.add_argument("-train", action='store_true', help="Execute LLM training (default: %(default)s)")
    parser.add_argument("-predict", action='store_true', help="Execute LLM prediction (default: %(default)s)")

    # Parse the command-line arguments
    args = parser.parse_args()
    return args


def main() -> None:
    """Main function to execute the script"""
    args = add_args()


    config = load_config()



    if args.menu:
        while True:
            menu(args, config)
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
            StockDNA.exec(StockPreprocessor.chunk_size, config)
        if args.train_preparation:
            TrainPreparation.exec(args)
        if args.master_data:
            master_data()
        if args.train:
            LLM.train()
        if args.predict:
            LLM.predict()
            LLM.true_predict()

    print("Done!")


def menu(args, config) -> None:
    """
    Displays a menu for users who run the script without CLI arguments
    """

    while True:
        print("\n\n" + "=" * 80 + "\n")
    
        print("1. Stock Symbol Verification")
        print("2. Stock Data Collection")
        print("3. Stock Preprocessor")
        print("4. Stock DNA")
        print("5. Train Preparation")
        print("6. Master Data")
        print("7. LLM")
        print("   7.1 LLM Training")
        print("   7.2 LLM Prediction")
        print("Q. Quit")
        print()

        choice = input("Choose an option (1-7, Q to quit): ")

        print()

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
            StockDNA.exec(StockPreprocessor.chunk_size, config)
        elif choice == '5':
            TrainPreparation.exec(args)
        elif choice == '6':
            master_data()
        elif choice == '7':  # Logic for LLM choices
            subchoice = input("   Choose an option (1 or 2): ")
            if subchoice == '1':
                LLM.train()
            elif subchoice == '2':
                LLM.predict()
                LLM.true_predict()

if __name__ == "__main__":
    main()
    
