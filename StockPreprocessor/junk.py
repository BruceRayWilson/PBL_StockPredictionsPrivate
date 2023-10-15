    def preprocess_file(self, file):
        # Read CSV file
        df = pd.read_csv(os.path.join(self.input_directory, file))

        # Keep only the necessary fields
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Calculate Tomorrow_Gain as tomorrow's close divided by tomorrow's open
        df['Tomorrow_Gain'] = df['Close'].shift(-1) / df['Open'].shift(-1)

        # Drop rows with NaN values
        df.dropna(inplace=True)

        # Divide data into chunks of the specified size in reverse order
        chunks = [df[i:i + self.chunk_size] for i in range(df.shape[0] - self.chunk_size, -1, -self.chunk_size)]

        # Initialize an empty DataFrame to store the concatenated chunks
        result_df = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Tomorrow_Gain', 'Volume'])

        for chunk in chunks:
            # Normalize 'Open', 'High', 'Low', 'Close', 'Tomorrow_Gain' as a group
            ohlc_data = chunk[['Open', 'High', 'Low', 'Close', 'Tomorrow_Gain']]
            volume_data = chunk[['Volume']]

            ohlc_scaler = MinMaxScaler()
            volume_scaler = MinMaxScaler()

            chunk.loc[:, ['Open', 'High', 'Low', 'Close', 'Tomorrow_Gain']] = ohlc_scaler.fit_transform(ohlc_data)
            chunk.loc[:, ['Volume']] = volume_scaler.fit_transform(volume_data)

            # Append the chunk to the result DataFrame
            result_df = pd.concat([result_df, chunk])

        # Sort the result DataFrame by 'Date'
        result_df.sort_values(by=['Date'], inplace=True)

        # Save the preprocessed data to a new CSV file
        result_df.to_csv(os.path.join(self.output_directory, file), index=False)
