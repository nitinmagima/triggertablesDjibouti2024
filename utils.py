# --------------------------------------------------------------------------------------------------
# Functions for Trigger Tables
#
# Author - Nitin Magima
# Date - 2024
# Version - 1.0
# --------------------------------------------------------------------------------------------------

# ==================================================================================================
#
# IMPORTANT - DISCLAIMER AND RIGHTS STATEMENT
# This is a set of scripts written by the Financial Instruments Team at the International Research
# Institute for Climate and Society (IRI) part of The Columbia Climate School, Columbia University
# They are shared for educational purposes only.  Anyone who uses this code or its
# functionality or structure, assumes full liability and should inform and credit IRI.
#
# ==================================================================================================

# Loading Packages
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import HTML
import yaml


def load_config(file_path="config.yaml"):
    """
    Loads configuration data from a YAML file.

    Args:
    - config_file (str): Path to the YAML configuration file.

    Returns:
    - dict: Dictionary containing configuration data.
    """    
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def get_data(maproom, mode, region, season, predictor, predictand, year, bad_years,
             issue_month0, freq, include_upcoming, threshold_protocol, username, password):
    """
    Retrieves data from an API endpoint and combines it into a DataFrame.
    ...
    """
    # Make a GET request to the API
    region_str = ",".join(map(str, region))  # Convert region values to a comma-separated string
    api_url = (f"http://iridl.ldeo.columbia.edu/fbfmaproom2/{maproom}/"
               f"export?season={season}&issue_month0={issue_month0}&freq={freq}&predictor"
               f"={predictor}&predictand={predictand}&include_upcoming={include_upcoming}&mode={mode}"
               f"&region={region_str}")

    # Constructing the design tool URL
    tool_url = (f"https://iridl.ldeo.columbia.edu/fbfmaproom2/{maproom}?mode={mode}&map_column={predictor}"
               f"&season={season}&predictors={predictor}&predictand={predictand}&year={year}"
               f"&issue_month0={issue_month0}&freq={freq}&severity=0&include_upcoming={include_upcoming}")

    auth = (username, password)
    response = requests.get(api_url, auth=auth)

    if response.status_code == 200:
        json_data = response.json()
        flattened_data = pd.json_normalize(json_data)

        non_nested_columns = []

        for column in flattened_data.columns:
            if isinstance(flattened_data[column][0], list):
                expanded_data = pd.json_normalize(flattened_data[column].explode(), sep='_')
                flattened_data = pd.concat([flattened_data, expanded_data], axis=1)
                flattened_data = flattened_data.drop(column, axis=1)
            else:
                non_nested_columns.append(column)

        non_nested_df = flattened_data[non_nested_columns]
        melted_non_nested_df = pd.DataFrame({
            'Metric': non_nested_df.columns,
            'Value': non_nested_df.iloc[0].values
        })

        replace_values = {
            'threshold': 'Forecast Threshold',
            'skill.accuracy': 'Forecast Accuracy',
            'skill.act_in_vain': 'Act in Vain',
            'skill.fail_to_act': 'Fail to Act',
            'skill.worthy_action': 'Worthy Action',
            'skill.worthy_inaction': 'Worthy Inaction'
        }
        melted_non_nested_df['Metric'] = melted_non_nested_df['Metric'].replace(replace_values)


        # Convert melted_non_nested_df to a dictionary
        melted_non_nested_dict = melted_non_nested_df.set_index('Metric')['Value'].to_dict()

        # Convert flattened data to Pandas DataFrame
        df = pd.DataFrame(flattened_data).drop(non_nested_df.columns, axis=1, errors='ignore')
        df['Triggered'] = df[predictor] > melted_non_nested_dict['Forecast Threshold']
        df['Trigger Difference'] = df[predictor] - melted_non_nested_dict['Forecast Threshold']
        df['Adjusted Forecast Threshold'] = melted_non_nested_dict['Forecast Threshold'] + threshold_protocol
        df['Triggered Adjusted'] = df[predictor] > melted_non_nested_dict['Forecast Threshold']
        df.rename(columns={predictor: 'Forecast', 'year': 'Year'}, inplace=True)

        # Filter df based on the provided list of years
        df = df[df['Year'].isin(bad_years)]
        
        # Select relevant columns including 'year', and no longer limiting the DataFrame to the first row
        df = df.loc[:, ['Year', 'Forecast', 'Trigger Difference', 'Triggered', 'Triggered Adjusted', 'Adjusted Forecast Threshold']]

        # Combine df and melted_non_nested_df
     
        combined_df = df

        melted_non_nested_df.set_index('Metric', inplace=True)

        combined_df['Act in Vain'] = melted_non_nested_df.at['Act in Vain', 'Value']
        combined_df['Fail to Act'] = melted_non_nested_df.at['Fail to Act', 'Value']
        combined_df['Worthy Action'] = melted_non_nested_df.at['Worthy Action', 'Value']
        combined_df['Worthy Inaction'] = melted_non_nested_df.at['Worthy Inaction', 'Value']
        combined_df['Frequency (%)'] = f"{freq}%"
        combined_df['Forecast Accuracy (%)'] = melted_non_nested_df.at['Forecast Accuracy', 'Value']
        combined_df['Forecast Threshold'] = melted_non_nested_df.at['Forecast Threshold', 'Value']
        combined_df['Threshold Protocol'] = f"{threshold_protocol}"

        month_mapping = {
            0: 'Jan',
            1: 'Feb',
            2: 'Mar',
            3: 'Apr',
            4: 'May',
            5: 'Jun',
            6: 'Jul',
            7: 'Aug',
            8: 'Sep',
            9: 'Oct',
            10: 'Nov',
            11: 'Dec'
        }

        combined_df['Issue Month'] = issue_month0
        combined_df['Issue Month'] = combined_df['Issue Month'].map(month_mapping)
        combined_df['Design Tool URL'] = f"<a href='{tool_url}'>Design Tool Link</a>"

        # Define the sequence of desired columns
        desired_columns = ['Year', 'Frequency (%)', 'Issue Month', 'Forecast', 'Forecast Threshold', 'Trigger Difference',
                           'Forecast Accuracy (%)', 'Triggered', 'Adjusted Forecast Threshold', 'Threshold Protocol',
                           'Triggered Adjusted', 'Act in Vain', 'Fail to Act', 'Worthy Action', 'Worthy Inaction',
                           'Design Tool URL']
        combined_df = combined_df.reindex(columns=desired_columns)

        combined_df = combined_df.rename(columns={
            'forecast': 'Forecast',
            'triggered': 'Triggered',
            'trigger difference': 'Trigger Difference'
            # Additional renaming handled by replace_values
        })

        return combined_df
    else:
        print(f"Error: {response.status_code}")
        return pd.DataFrame()



def get_admin_data(maproom, level, username, password, need_valid_keys, valid_keys=None):
    """
    Retrieves administrative data from an API endpoint.

    Args:
    - maproom (str): Maproom value.
    - level (str): Level of administrative data.

    Returns:
    - DataFrame: DataFrame containing administrative data.
    """
    # Construct the API URL with the provided parameters
    api_url = f"http://iridl.ldeo.columbia.edu/fbfmaproom2/regions?country={maproom}&level={level}"

    # Make a GET request to the API
    if username and password:
        auth = (username, password)
        response = requests.get(api_url, auth=auth)
    else:
        response = requests.get(api_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON data
        json_data = response.json()

        # Create a DataFrame from the JSON data
        df = pd.DataFrame(json_data)

        # Extract "key" and "label" from the "regions" column
        df[['key', 'label']] = df['regions'].apply(pd.Series)

        # Filter keys if valid_keys is provided
        if level != 0:
            if need_valid_keys is True:
                df = df[df['key'].isin(valid_keys)]

        # Drop the original "regions" column if needed
        df = df.drop('regions', axis=1)

        return df
    else:
        # Print an error message if the request was not successful
        print(f"Error: {response.status_code}")
        return None


def get_trigger_tables(maproom, mode, season, predictor, predictand, year, bad_years,
                       issue_month, frequencies, include_upcoming, threshold_protocol, username, password,
                       need_valid_keys, valid_keys):
    """
    Retrieves trigger tables based on specified parameters.

    Args:
    - maproom (str): Maproom value.
    - mode (int): Mode value.
    - season (str): Season value.
    - predictor (str): Predictor value.
    - predictand (str): Predictand value.
    - issue_month (list): List of issue month values.
    - frequencies (list): List of frequency values.
    - include_upcoming (str): Include upcoming value.
    - threshold_protocol (int): Threshold protocol value.
    - username (str): Username for API authentication.
    - password (str): Password for API authentication.
    - need_valid_keys (bool): Flag indicating if valid keys are needed.
    - valid_keys (list): List of valid keys.

    Returns:
    - dict: Dictionary containing trigger tables.
    """
    print("Fetching....")
    # Initialize a dictionary to store admin tables
    admin_tables = {}

    # Creating trigger tables

    admin_name = f"admin{mode}_tables"
    admin_tables[admin_name] = {}
    admin_data = get_admin_data(maproom, mode, username=username, password=password,
                                need_valid_keys=need_valid_keys, valid_keys=valid_keys)

    for freq in frequencies:
        for month in issue_month:
            # Iterate over each key value
            if isinstance(admin_data, pd.Series):
                for region_key, label in admin_data.items():
                    print(region_key, label)
                    table_name = f"output_freq_{freq}_mode_{mode}_month_{month}_region_{region_key}_table"

                    df = get_data(maproom=maproom, mode=mode, region=[region_key],
                                  season=season, predictor=predictor, predictand=predictand, year = year,
                                  issue_month0=month, freq=freq, include_upcoming=include_upcoming,
                                  bad_years = bad_years,
                                  threshold_protocol=threshold_protocol, username=username, password=password)

                    df.insert(0, 'Admin Name', label)
                    admin_tables[admin_name][table_name] = df

            elif isinstance(admin_data, pd.DataFrame):
                for index, row in admin_data.iterrows():
                    region_key, label = row['key'], row['label']

                    table_name = f"output_freq_{freq}_mode_{mode}_month_{month}_region_{region_key}_table"

                    df = get_data(maproom=maproom, mode=mode, region=[region_key],
                                  season=season, predictor=predictor, predictand=predictand, year = year,
                                  issue_month0=month, freq=freq, include_upcoming=include_upcoming,
                                  bad_years = bad_years,
                                  threshold_protocol=threshold_protocol, username=username, password=password)

                    df.insert(0, 'Admin Name', label)
                    admin_tables[admin_name][table_name] = df

            else:
                # Handle other cases or raise an error
                raise ValueError("Unexpected output type from get_admin_data.")

    return admin_tables

def generate_colors(n):
    """
    Generates a list of n distinct colors in HSL format.
    
    Args:
        n (int): The number of distinct colors to generate.
        
    Returns:
        List[str]: A list of colors.
    """
    return [f"hsl({int((360 / n) * i)}, 100%, 70%)" for i in range(n)]

def style_and_render_df_with_hyperlinks(df):
    # Define columns to style
    columns_to_style = ['Admin Name', 'Severity', 'Frequency (%)', 'Issue Month']
    
    # Calculate the total number of unique values across all columns
    unique_values_count = sum(df[col].nunique() for col in columns_to_style)
    
    # Generate a unique color for each unique value across all columns
    unique_colors = generate_colors(unique_values_count)
    
    # Assign a distinct segment of colors to each column
    color_index = 0
    color_maps = {}
    for column in columns_to_style:
        unique_values = df[column].unique()
        color_maps[column] = {value: unique_colors[color_index + i] for i, value in enumerate(unique_values)}
        color_index += len(unique_values)
    
    # Function to apply colors based on the value for a given column
    def apply_color(val, column):
        if pd.isnull(val):
            return ''  # Return default style for NaN values
        return f'background-color: {color_maps[column].get(val, "")};'
    
    # Initialize the styled DataFrame
    styled_df = df.style
    
    # Apply the styles to each column individually
    for column in columns_to_style:
        styled_df = styled_df.map(lambda val, col=column: apply_color(val, col), subset=[column])
    
    # Apply boolean highlights for 'triggered' and 'Triggered Adjusted' columns
    true_color, false_color = '#CCFFCC', '#FFCC99'
    
    # Assuming styled_df is your DataFrame styled object, and true_color/false_color are defined
    columns_to_style = ['Triggered', 'Triggered Adjusted']
    
    for col in columns_to_style:
        try:
            # Check if column exists by trying to access it
            if col in styled_df.columns:
                # Apply the styling if column exists
                styled_df = styled_df.map(lambda val: f'background-color: {true_color if val else false_color}', subset=[col])
            else:
                # If the column does not exist in the DataFrame, this line will not be executed
                pass
        except KeyError as e:
            print(f"Column not found: {e}")
            # Handle the case where the column does not exist, e.g., by logging or passing
            continue
    
    # Format numerical columns
    styled_df = styled_df.format({'Forecast': "{:.2f}", 'Trigger Difference': "{:.2f}", 'Forecast Accuracy (%)': "{:.2%}",'Forecast Threshold': "{:.2f}", 'Act in Vain': "{:.1f}", 'Fail to Act': "{:.1f}", 'Worthy Action': "{:.1f}", 'Worthy Inaction': "{:.1f}", 'Adjusted Forecast Threshold': "{:.2f}" })
    
    # Render to HTML
    rendered_html = styled_df.to_html(escape=False)
    display(HTML(rendered_html))