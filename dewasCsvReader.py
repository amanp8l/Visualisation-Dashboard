import os
import pandas as pd
from openai import AzureOpenAI
import io
import contextlib
import matplotlib
import matplotlib.pyplot as plt
import json
from datetime import datetime
import traceback
import base64
import warnings
import re
from mimetypes import guess_type
from dotenv import load_dotenv
import numpy as np

# Load environment variables from .env file
load_dotenv()

conversationalMessage = []
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-10-21",
)

def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def get_excel_sheets_info(file_path):
    """Get information about all sheets in an Excel file"""
    if file_path.lower().endswith('.xlsx') or file_path.lower().endswith('.xls'):
        # Get sheet names
        xl = pd.ExcelFile(file_path)
        sheet_names = xl.sheet_names
        
        # Collect information about each sheet
        sheets_info = []
        for sheet in sheet_names:
            sheet_data = pd.read_excel(file_path, sheet_name=sheet, nrows=5)  # Read just a few rows for preview
            sheets_info.append({
                'name': sheet,
                'columns': list(sheet_data.columns),
                'sample_rows': min(5, len(sheet_data)),
                'shape': pd.read_excel(file_path, sheet_name=sheet).shape
            })
        return sheets_info
    return None

def get_data_insights(file_path, file_info=None):
    """Extract insights about data including common tags in columns"""
    insights = {"common_tags": {}, "data_types": {}, "value_ranges": {}}
    
    if file_path.lower().endswith('.csv'):
        df = pd.read_csv(file_path)
        sheets = [{"name": "main", "df": df}]
    elif file_path.lower().endswith(('.xlsx', '.xls')):
        # Read all sheets
        sheets = []
        excel_file = pd.ExcelFile(file_path)
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            sheets.append({"name": sheet_name, "df": df})
    
    # Process each sheet/dataframe
    for sheet in sheets:
        sheet_name = sheet["name"]
        df = sheet["df"]
        
        # Look for common tags in column names
        column_words = {}
        for col in df.columns:
            # Split column names into words
            words = re.findall(r'[A-Za-z0-9]+', col)
            for word in words:
                if len(word) > 2:  # Ignore very short words
                    word = word.lower()
                    if word not in column_words:
                        column_words[word] = []
                    column_words[word].append(col)
        
        # Filter for words that appear in multiple columns (potential tags)
        for word, columns in column_words.items():
            if len(columns) > 1:
                if word not in insights["common_tags"]:
                    insights["common_tags"][word] = {}
                insights["common_tags"][word][sheet_name] = columns
        
        # Data types and basic stats for each column
        sheet_insights = {}
        for col in df.columns:
            col_type = str(df[col].dtype)
            
            # Get value ranges for numeric and datetime columns
            if col_type.startswith('int') or col_type.startswith('float'):
                try:
                    sheet_insights[col] = {
                        "type": col_type,
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                        "mean": float(df[col].mean()),
                        "null_count": int(df[col].isna().sum())
                    }
                except:
                    sheet_insights[col] = {"type": col_type, "error": "Could not compute stats"}
            elif col_type.startswith('datetime'):
                try:
                    sheet_insights[col] = {
                        "type": col_type,
                        "min": str(df[col].min()),
                        "max": str(df[col].max()),
                        "null_count": int(df[col].isna().sum())
                    }
                except:
                    sheet_insights[col] = {"type": col_type, "error": "Could not compute stats"}
            else:
                # For categorical/text columns, get unique values count
                try:
                    unique_count = df[col].nunique()
                    sheet_insights[col] = {
                        "type": col_type,
                        "unique_values": int(unique_count),
                        "null_count": int(df[col].isna().sum()),
                        "sample_values": df[col].dropna().unique()[:5].tolist() if unique_count < 50 else "Too many to list"
                    }
                except:
                    sheet_insights[col] = {"type": col_type, "error": "Could not compute stats"}
        
        insights["data_types"][sheet_name] = sheet_insights
    
    return insights

def generate_chart_data(file_path, chart_type="overview"):
    """Generate chart data for frontend visualization based on chart type"""
    try:
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
            sheet_name = "main"
        elif file_path.lower().endswith(('.xlsx', '.xls')):
            # Use first sheet as default
            df = pd.read_excel(file_path, engine='openpyxl')
            sheet_name = pd.ExcelFile(file_path, engine='openpyxl').sheet_names[0]
        else:
            return None
        
        # Sample or process data based on size
        if len(df) > 1000:
            df = df.sample(1000, random_state=42)  # Sample for large datasets
        
        chart_config = {}
        
        if chart_type == "overview":
            # Bar chart for top 5-10 numeric columns by mean value
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) > 0:
                # Take up to 7 numeric columns
                cols_to_use = numeric_cols[:7]
                means = df[cols_to_use].mean().sort_values(ascending=False)
                
                chart_config = {
                    "type": "bar",
                    "data": {
                        "labels": means.index.tolist(),
                        "datasets": [{
                            "label": "Average Values",
                            "data": means.values.tolist(),
                            "backgroundColor": [
                                'rgba(67, 97, 238, 0.7)',
                                'rgba(247, 37, 133, 0.7)',
                                'rgba(58, 12, 163, 0.7)', 
                                'rgba(46, 196, 182, 0.7)',
                                'rgba(255, 159, 28, 0.7)',
                                'rgba(255, 99, 132, 0.7)',
                                'rgba(54, 162, 235, 0.7)'
                            ],
                            "borderColor": "rgba(67, 97, 238, 1)",
                            "borderWidth": 1
                        }]
                    }
                }
            else:
                # Fallback to categorical data if no numeric columns
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                if len(categorical_cols) > 0:
                    col = categorical_cols[0]
                    counts = df[col].value_counts().head(7)
                    
                    chart_config = {
                        "type": "pie",
                        "data": {
                            "labels": counts.index.tolist(),
                            "datasets": [{
                                "data": counts.values.tolist(),
                                "backgroundColor": [
                                    'rgba(67, 97, 238, 0.7)',
                                    'rgba(247, 37, 133, 0.7)',
                                    'rgba(58, 12, 163, 0.7)', 
                                    'rgba(46, 196, 182, 0.7)',
                                    'rgba(255, 159, 28, 0.7)',
                                    'rgba(255, 99, 132, 0.7)',
                                    'rgba(54, 162, 235, 0.7)'
                                ],
                                "borderWidth": 1
                            }]
                        }
                    }
        
        elif chart_type == "trend":
            # Find datetime or sequential columns for trend
            date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if len(date_cols) > 0 and len(numeric_cols) > 0:
                # Use first date column and first numeric column
                date_col = date_cols[0]
                numeric_col = numeric_cols[0]
                
                # Group by date and aggregate
                trend_data = df.groupby(pd.Grouper(key=date_col, freq='M'))[numeric_col].mean().reset_index()
                trend_data = trend_data.sort_values(by=date_col)
                
                chart_config = {
                    "type": "line",
                    "data": {
                        "labels": [d.strftime('%b %Y') for d in trend_data[date_col]],
                        "datasets": [{
                            "label": f"{numeric_col} Over Time",
                            "data": trend_data[numeric_col].tolist(),
                            "borderColor": "rgba(67, 97, 238, 1)",
                            "backgroundColor": "rgba(67, 97, 238, 0.2)",
                            "tension": 0.4,
                            "fill": True
                        }]
                    }
                }
            elif len(numeric_cols) >= 2:
                # Use two numeric columns if no date column
                x_col = numeric_cols[0]
                y_col = numeric_cols[1]
                
                chart_config = {
                    "type": "line",
                    "data": {
                        "labels": [str(i) for i in range(min(30, len(df)))],
                        "datasets": [{
                            "label": y_col,
                            "data": df[y_col].head(30).tolist(),
                            "borderColor": "rgba(247, 37, 133, 1)",
                            "backgroundColor": "rgba(247, 37, 133, 0.2)",
                            "tension": 0.4,
                            "fill": True
                        }]
                    }
                }
        
        elif chart_type == "distribution":
            # Histogram for numeric data distribution
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) > 0:
                # Use first numeric column
                col = numeric_cols[0]
                
                # Calculate histogram data
                hist, bins = np.histogram(df[col].dropna(), bins=7)
                bin_labels = [f"{round(bins[i], 2)}-{round(bins[i+1], 2)}" for i in range(len(bins)-1)]
                
                chart_config = {
                    "type": "bar",
                    "data": {
                        "labels": bin_labels,
                        "datasets": [{
                            "label": f"Distribution of {col}",
                            "data": hist.tolist(),
                            "backgroundColor": "rgba(58, 12, 163, 0.7)",
                            "borderColor": "rgba(58, 12, 163, 1)",
                            "borderWidth": 1
                        }]
                    }
                }
            else:
                # Fallback to categorical distribution
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                if len(categorical_cols) > 0:
                    col = categorical_cols[0]
                    counts = df[col].value_counts().head(7)
                    
                    chart_config = {
                        "type": "doughnut",
                        "data": {
                            "labels": counts.index.tolist(),
                            "datasets": [{
                                "data": counts.values.tolist(),
                                "backgroundColor": [
                                    'rgba(67, 97, 238, 0.7)',
                                    'rgba(247, 37, 133, 0.7)',
                                    'rgba(58, 12, 163, 0.7)', 
                                    'rgba(46, 196, 182, 0.7)',
                                    'rgba(255, 159, 28, 0.7)',
                                    'rgba(255, 99, 132, 0.7)',
                                    'rgba(54, 162, 235, 0.7)'
                                ],
                                "borderWidth": 1
                            }]
                        }
                    }
        
        elif chart_type == "correlation":
            # Scatter plot for correlation between two numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) >= 2:
                col1 = numeric_cols[0]
                col2 = numeric_cols[1]
                
                # Sample data for scatter plot (max 100 points)
                sample_size = min(100, len(df))
                sample_df = df.sample(sample_size)
                
                chart_config = {
                    "type": "scatter",
                    "data": {
                        "datasets": [{
                            "label": f"{col1} vs {col2}",
                            "data": [{"x": x, "y": y} for x, y in zip(sample_df[col1], sample_df[col2])],
                            "backgroundColor": "rgba(46, 196, 182, 0.7)",
                            "borderColor": "rgba(46, 196, 182, 1)",
                            "pointRadius": 6,
                            "pointHoverRadius": 8
                        }]
                    }
                }
            elif len(numeric_cols) > 0:
                # Fallback to single numeric column distribution
                col = numeric_cols[0]
                hist, bins = np.histogram(df[col].dropna(), bins=7)
                bin_labels = [f"{round(bins[i], 2)}-{round(bins[i+1], 2)}" for i in range(len(bins)-1)]
                
                chart_config = {
                    "type": "bar",
                    "data": {
                        "labels": bin_labels,
                        "datasets": [{
                            "label": f"Distribution of {col}",
                            "data": hist.tolist(),
                            "backgroundColor": "rgba(46, 196, 182, 0.7)",
                            "borderColor": "rgba(46, 196, 182, 1)",
                            "borderWidth": 1
                        }]
                    }
                }
                
        elif chart_type == "custom":
            # Radar chart for comparing multiple metrics
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) >= 3:
                # Use up to 6 numeric columns
                cols_to_use = numeric_cols[:6]
                
                # Normalize values for radar chart
                normalized_values = []
                for col in cols_to_use:
                    # Get mean or other representative value
                    val = df[col].mean()
                    # Min-max normalize to 0-100 range
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if max_val > min_val:
                        normalized = (val - min_val) / (max_val - min_val) * 100
                    else:
                        normalized = 50  # Default value if min equals max
                    normalized_values.append(normalized)
                
                chart_config = {
                    "type": "radar",
                    "data": {
                        "labels": cols_to_use,
                        "datasets": [{
                            "label": "Data Profile",
                            "data": normalized_values,
                            "backgroundColor": "rgba(255, 159, 28, 0.2)",
                            "borderColor": "rgba(255, 159, 28, 1)",
                            "borderWidth": 2,
                            "pointBackgroundColor": "rgba(255, 159, 28, 1)",
                            "pointBorderColor": "#fff",
                            "pointHoverBackgroundColor": "#fff",
                            "pointHoverBorderColor": "rgba(255, 159, 28, 1)"
                        }]
                    }
                }
            else:
                # Fallback to bar chart
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                if len(categorical_cols) > 0:
                    col = categorical_cols[0]
                    counts = df[col].value_counts().head(7)
                    
                    chart_config = {
                        "type": "bar",
                        "data": {
                            "labels": counts.index.tolist(),
                            "datasets": [{
                                "label": f"Count of {col}",
                                "data": counts.values.tolist(),
                                "backgroundColor": "rgba(255, 159, 28, 0.7)",
                                "borderColor": "rgba(255, 159, 28, 1)",
                                "borderWidth": 1
                            }]
                        }
                    }
        
        return {
            "chart_config": chart_config,
            "sheet_name": sheet_name
        }
    
    except Exception as e:
        print(f"Error generating chart data: {str(e)}")
        traceback.print_exc()
        return None

def generate_file_summary(file_path):
    """Generate summary information about the file for the dashboard"""
    try:
        file_info = {'file_type': None, 'sheets_info': None}
        summary_data = {}
        
        # Get basic file info
        file_size = os.path.getsize(file_path)
        if file_size < 1024:
            file_size_str = f"{file_size} bytes"
        elif file_size < 1024 * 1024:
            file_size_str = f"{file_size/1024:.1f} KB"
        else:
            file_size_str = f"{file_size/(1024*1024):.1f} MB"
        
        summary_data['fileSize'] = file_size_str
        summary_data['fileName'] = os.path.basename(file_path)
        
        # Handle different file types
        if file_path.lower().endswith('.csv'):
            file_info['file_type'] = 'csv'
            df = pd.read_csv(file_path)
            summary_data['fileType'] = 'CSV'
            summary_data['rows'] = len(df)
            summary_data['columns'] = len(df.columns)
            
            # Get sheet info (just one sheet for CSV)
            sheet_data = {
                'name': 'main',
                'rowCount': len(df),
                'columnCount': len(df.columns),
                'columns': list(df.columns)
            }
            summary_data['sheets'] = [sheet_data]
            
        elif file_path.lower().endswith(('.xlsx', '.xls')):
            file_info['file_type'] = 'excel'
            # Get information about all sheets
            file_info['sheets_info'] = get_excel_sheets_info(file_path)
            
            summary_data['fileType'] = 'Excel'
            summary_data['sheets'] = []
            
            total_rows = 0
            max_columns = 0
            
            for sheet_info in file_info['sheets_info']:
                sheet_name = sheet_info['name']
                row_count = sheet_info['shape'][0]
                col_count = sheet_info['shape'][1]
                
                total_rows += row_count
                max_columns = max(max_columns, col_count)
                
                sheet_data = {
                    'name': sheet_name,
                    'rowCount': row_count,
                    'columnCount': col_count,
                    'columns': sheet_info['columns']
                }
                summary_data['sheets'].append(sheet_data)
            
            summary_data['rows'] = total_rows
            summary_data['columns'] = max_columns
        
        # Extract data insights including common tags
        file_info['data_insights'] = get_data_insights(file_path, file_info)
        
        # Get key metrics
        key_metrics = []
        quick_stats = []
        
        if file_info['file_type'] == 'csv':
            dfs = [('main', pd.read_csv(file_path))]
        else:
            dfs = []
            for sheet in file_info['sheets_info']:
                df = pd.read_excel(file_path, sheet_name=sheet['name'], engine='openpyxl')
                dfs.append((sheet['name'], df))
        
        # Process each dataframe
        for sheet_name, df in dfs:
            # Get numeric columns for stats
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            # Add null value statistics
            null_count = df.isna().sum().sum()
            null_percent = (null_count / (df.shape[0] * df.shape[1]) * 100) if df.size > 0 else 0
            key_metrics.append({
                'name': f'Missing Values{" in " + sheet_name if len(dfs) > 1 else ""}',
                'value': f'{null_count} ({null_percent:.1f}%)'
            })
            
            # Add basic stats for numeric columns
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                try:
                    key_metrics.append({
                        'name': f'Avg {col}{" in " + sheet_name if len(dfs) > 1 else ""}',
                        'value': f'{df[col].mean():.2f}'
                    })
                    
                    if len(quick_stats) < 4:  # Add only up to 4 quick stats
                        quick_stats.append({
                            'name': f'{col} (Max)',
                            'value': f'{df[col].max():.0f}'
                        })
                except:
                    pass
            
            # Add categorical column stats
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            for col in cat_cols[:2]:  # Limit to first 2 categorical columns
                try:
                    unique_count = df[col].nunique()
                    key_metrics.append({
                        'name': f'Unique {col}{" in " + sheet_name if len(dfs) > 1 else ""}',
                        'value': f'{unique_count}'
                    })
                    
                    # Add most common value if not already reached limit
                    if len(quick_stats) < 4:
                        most_common = df[col].value_counts().index[0]
                        if len(str(most_common)) > 15:
                            most_common = str(most_common)[:12] + "..."
                        quick_stats.append({
                            'name': f'Most Common {col}',
                            'value': f'{most_common}'
                        })
                except:
                    pass
        
        # Use GPT to generate a summary of the file
        summary_text = generate_ai_summary(file_path, file_info)
        summary_data['summary'] = summary_text
        summary_data['keyMetrics'] = key_metrics
        summary_data['quickStats'] = quick_stats
        
        # Generate suggested visualizations
        summary_data['suggestedVisualizations'] = generate_suggested_visualizations(file_path, file_info)
        
        return summary_data
    
    except Exception as e:
        print(f"Error generating file summary: {str(e)}")
        traceback.print_exc()
        return {
            'summary': f"Error analyzing file: {str(e)}",
            'rows': 0,
            'columns': 0,
            'fileSize': 'Unknown',
            'fileName': os.path.basename(file_path),
            'fileType': 'Unknown'
        }

def generate_ai_summary(file_path, file_info):
    """Generate an AI-powered summary of the file contents"""
    try:
        # Construct file info for prompting
        file_details = f"File: {os.path.basename(file_path)}"
        
        if file_info['file_type'] == 'csv':
            df = pd.read_csv(file_path)
            file_details += f"\nType: CSV file with {len(df)} rows and {len(df.columns)} columns"
            file_details += f"\nColumns: {', '.join(df.columns.tolist())}"
            
            # Add sample data
            file_details += "\n\nSample data (first 5 rows):\n"
            file_details += df.head().to_string()
            
        elif file_info['file_type'] == 'excel':
            file_details += f"\nType: Excel file with {len(file_info['sheets_info'])} sheets"
            
            for sheet in file_info['sheets_info']:
                file_details += f"\n\nSheet: {sheet['name']}"
                file_details += f"\nRows: {sheet['shape'][0]}, Columns: {sheet['shape'][1]}"
                file_details += f"\nColumns: {', '.join(sheet['columns'])}"
        
        prompt = f"""Analyze this data file and provide a concise summary (2-3 sentences) that describes:
1. The type of data contained in the file
2. The main entities or subjects represented
3. Key variables/columns of interest
4. Potential analytical value of this dataset

File details:
{file_details}

Keep your summary focused, informative, and under 100 words.
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            max_tokens=150,
            messages=[
                {"role": "system", "content": "You are a data analyst who provides concise, insightful summaries of data files."},
                {"role": "user", "content": prompt}
            ]
        )
        
        summary = response.choices[0].message.content.strip()
        return summary
    
    except Exception as e:
        print(f"Error generating AI summary: {str(e)}")
        return "This file contains data that can be analyzed for insights. Upload complete."

def generate_suggested_visualizations(file_path, file_info):
    """Generate suggested visualizations for the dashboard"""
    try:
        visualizations = []
        
        # Read data based on file type
        if file_info['file_type'] == 'csv':
            dfs = [('main', pd.read_csv(file_path))]
        else:
            dfs = []
            for sheet in file_info['sheets_info'][:2]:  # Limit to first 2 sheets
                df = pd.read_excel(file_path, sheet_name=sheet['name'], engine='openpyxl')
                dfs.append((sheet['name'], df))
        
        for sheet_name, df in dfs:
            # Sample data if large
            if len(df) > 1000:
                df = df.sample(1000, random_state=42)
            
            # Check for datetime columns for time series
            date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            # Also check for potential date columns stored as strings
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    # Try to convert to datetime
                    pd.to_datetime(df[col], errors='raise')
                    date_cols.append(col)
                except:
                    pass
            
            # Check for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            # Time series visualization if we have date and numeric columns
            if date_cols and numeric_cols and len(visualizations) < 3:
                date_col = date_cols[0]
                numeric_col = numeric_cols[0]
                
                visualizations.append({
                    'title': f'Time Trend of {numeric_col}',
                    'type': 'line',
                    'sheet': sheet_name,
                    'description': f'Trend of {numeric_col} over time'
                })
            
            # Distribution of a numeric column
            if numeric_cols and len(visualizations) < 3:
                col = numeric_cols[0]
                
                visualizations.append({
                    'title': f'Distribution of {col}',
                    'type': 'histogram',
                    'sheet': sheet_name,
                    'description': f'Frequency distribution of {col} values'
                })
            
            # Categorical data breakdown
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            if cat_cols and len(visualizations) < 3:
                col = cat_cols[0]
                
                visualizations.append({
                    'title': f'Breakdown by {col}',
                    'type': 'pie',
                    'sheet': sheet_name,
                    'description': f'Distribution of data across {col} categories'
                })
            
            # Correlation between two numeric columns
            if len(numeric_cols) >= 2 and len(visualizations) < 3:
                col1 = numeric_cols[0]
                col2 = numeric_cols[1]
                
                visualizations.append({
                    'title': f'Correlation: {col1} vs {col2}',
                    'type': 'scatter',
                    'sheet': sheet_name,
                    'description': f'Relationship between {col1} and {col2}'
                })
        
        return visualizations
    
    except Exception as e:
        print(f"Error generating suggested visualizations: {str(e)}")
        return [
            {'title': 'Data Overview', 'type': 'bar', 'description': 'Overview of main metrics'},
            {'title': 'Distribution Analysis', 'type': 'pie', 'description': 'Distribution of key categories'},
            {'title': 'Trend Analysis', 'type': 'line', 'description': 'Trends over time'}
        ]

def split_complex_query(user_query, file_path, file_info):
    """Split a complex query into multiple simpler queries"""
    system_content = f'''You are a data analyst assistant specializing in breaking down complex data queries into simpler sub-queries.
    
Your task is to analyze this complex query and break it down into separate, standalone sub-queries that can be processed individually.
Each sub-query should focus on a specific analytical question that can be answered with a single visualization or analysis.

File information:
- File path: {file_path}
'''

    # Add multi-sheet information if available
    if file_info and file_info.get('sheets_info'):
        sheets_info = file_info['sheets_info']
        sheets_text = "This Excel file contains multiple sheets:\n"
        for idx, sheet in enumerate(sheets_info):
            sheets_text += f"Sheet {idx+1}: '{sheet['name']}' - {sheet['shape'][0]} rows, {sheet['shape'][1]} columns, columns: {sheet['columns']}\n"
        system_content += f"\n{sheets_text}\n"
    
    if file_info and file_info.get('data_insights'):
        insights = file_info['data_insights']
        if insights.get('common_tags'):
            tags_text = "\nCommon tags/themes found in column names:\n"
            for tag, sheet_columns in insights['common_tags'].items():
                tags_text += f"- '{tag}' appears in: "
                for sheet, columns in sheet_columns.items():
                    tags_text += f"\n  Sheet '{sheet}': {', '.join(columns)}"
                tags_text += "\n"
            system_content += tags_text
    
    system_content += '''
For the query provided by the user:

1. Determine if it contains multiple distinct analytical questions
2. If it does, split it into separate sub-queries
3. Return a JSON response with:
   - "is_complex": true/false
   - "sub_queries": [list of individual queries]
   - "reasoning": brief explanation of your reasoning

If it's a simple query that should be processed as a single analysis, set is_complex to false and put the original query in sub_queries.

IMPORTANT: Each sub-query should be self-contained and clearly ask for a specific analysis or visualization.
'''

    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_query}
        ]
    )
    
    try:
        result = json.loads(response.choices[0].message.content)
        return result
    except:
        # If parsing fails, return a default response
        return {
            "is_complex": False,
            "sub_queries": [user_query],
            "reasoning": "Using original query as fallback"
        }

def pythonCodeGenerator(user_query, filePath, columnsName, fileshape, file_info=None):
    system_content = f'''You have a file at {filePath},
columns are {columnsName} and shape is {fileshape}.

IMPORTANT: When working with Excel files:
1. ALWAYS specify the engine parameter when reading Excel files: pd.read_excel(..., engine='openpyxl')
2. Similarly use: pd.ExcelFile(..., engine='openpyxl')
'''

    # Add multi-sheet information if available
    if file_info and file_info.get('sheets_info'):
        sheets_info = file_info['sheets_info']
        sheets_text = "This Excel file contains multiple sheets:\n"
        for idx, sheet in enumerate(sheets_info):
            sheets_text += f"Sheet {idx+1}: '{sheet['name']}' - {sheet['shape'][0]} rows, {sheet['shape'][1]} columns, columns: {sheet['columns']}\n"
        system_content += f"\n{sheets_text}\n"
        system_content += "When working with this Excel file, please check which sheet the user is interested in or analyze all sheets if needed.\n"
    
    # Add tag information if available
    if file_info and file_info.get('data_insights') and file_info['data_insights'].get('common_tags'):
        tags = file_info['data_insights']['common_tags']
        if tags:
            tags_text = "\nCommon tags found in column names that might be useful for analysis:\n"
            for tag, sheet_columns in tags.items():
                tags_text += f"- '{tag}' appears in: "
                for sheet, columns in sheet_columns.items():
                    tags_text += f"\n  Sheet '{sheet}': {', '.join(columns)}"
                tags_text += "\n"
            system_content += tags_text

    system_content += '''
Understand the User question and give response in json.
*user question may or may not be related to the file; answer accordingly.
*response json has two key 'response' and 'code'.

For questions asking about columns, sheets, or basic file information:
1. If the user is asking about the columns in all sheets, sheet names, or basic structure:
   - Put a well-formatted, human-readable response in the 'response' key
   - Format column lists with bullet points for readability
   - For each sheet, include heading and list columns in a clear, readable format
   - Make 'code' key null
2. Otherwise, write code to perform the requested operation in the 'code' key and make 'response' key null

For Excel files with multiple sheets:
1. If the user specifies a sheet, use that sheet.
2. If not specified but can be inferred from the query, use the most relevant sheet.
3. If it's unclear, include code to either list all sheets or analyze all/key sheets.
4. ALWAYS specify engine='openpyxl' when using pd.read_excel or pd.ExcelFile

For visualization:
1. ALWAYS create informative visualizations with proper labels, titles, and legends
2. Use appropriate plot types based on the data and analysis needed
3. For comparisons, use bar charts, line charts, or scatter plots as appropriate
4. Include comprehensive explanations in chart titles and annotations

Example responses:
user:"what sheets are in this file?"
assistant:{'response':"This Excel file contains the following sheets: 'Sales', 'Inventory', 'Customers'.",'code':null}

user:"what are the columns in all sheets"
assistant:{'response':"Here are the columns in each sheet of this Excel file:\n\n**Sheet: Sales**\n- Product ID\n- Date\n- Quantity\n- Revenue\n- Customer ID\n\n**Sheet: Inventory**\n- Product ID\n- Stock Level\n- Reorder Point\n- Last Updated\n\n**Sheet: Customers**\n- Customer ID\n- Name\n- Email\n- Location\n- Signup Date",'code':null}

user:"show me the first 5 rows from the Sales sheet"
assistant:{'response':null,'code':"import pandas as pd\n\ndf = pd.read_excel('path_to_file.xlsx', sheet_name='Sales', engine='openpyxl')\nprint(df.head())"}

*response are not null when answer already present in the context or previous conversation.
*code always load the data and import necessary library.
*for plotting YOU MUST USE ONLY matplotlib and seaborn library.
*ALL FILE PATHS must use forward slashes (/) or raw strings with r prefix to avoid unicode escape errors.
*don't tell about file path and name in response key.
*while generating code please keep proper indentation. because 'code' key value will be directly executing'''

    messages = [{"role": "system", "content": system_content}]
    messages.extend(user_query)
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        temperature=0,
        messages=messages
    )
    print(response.usage.total_tokens)
    print("pythonCodeGenerator:", response.choices[0].message.content)
    return response.choices[0].message.content

def codeExecutor(code):
    # Set the backend to 'Agg' for headless rendering
    print("generated code: \n", code)
    matplotlib.use('Agg')
    filename = datetime.now().strftime("%Y%m%d%H%M%S") + ".png"
    # Create a dictionary to serve as the local execution context
    exec_globals = {}
    
    # Capture standard output and standard error
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    errors = None
    
    # Suppress warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # This will ignore all warnings
        try:
            plt.close('all')
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                # Check if code contains pd.ExcelFile or pd.read_excel without engine
                if ("pd.ExcelFile" in code or "pd.read_excel" in code) and "engine=" not in code:
                    # Add engine parameter to Excel-related functions
                    code = code.replace("pd.ExcelFile(", "pd.ExcelFile(engine='openpyxl', ")
                    code = re.sub(r'pd\.read_excel\(([^,\)]+)', r'pd.read_excel(\1, engine="openpyxl"', code)
                
                # Execute the modified code string
                exec(code, exec_globals)

                # After executing `plt.show()`, capture the current figure
                fig = plt.gcf()

                # Check if the figure has any axes (i.e., if a plot was created)
                if fig.get_axes():  # Checks if there are any axes (i.e., plot)
                    # Create a BytesIO buffer to save the plot
                    plot_buffer = io.BytesIO()
                    fig.savefig(plot_buffer, format='png', dpi=100, bbox_inches='tight')
                    plot_buffer.seek(0)  # Rewind the buffer to the beginning

                    # Save the captured image directly
                    with open("public/" + filename, "wb") as f:
                        f.write(plot_buffer.getvalue())
                else:
                    filename = None

        except Exception as e:
            errors = traceback.format_exc()
            print(f"An error occurred: {e}")
            stderr_buffer.write(errors)

    # Get the captured stdout and stderr
    output = stdout_buffer.getvalue()
    errors = stderr_buffer.getvalue()

    # Print the captured outputs and errors
    print("Output:")
    print(output)
    print("Errors:")
    print(errors)
    if errors == "":
        errors = None
    return output, errors, filename

def pythonErrorResolver(error, user_query, code, columnsName, filePath, file_info=None):
    # Fix file path in code if there's a unicode escape error
    if "unicodeescape codec can't decode bytes" in error:
        # Replace backslashes with forward slashes or use raw string prefix
        code = code.replace(filePath, filePath.replace('\\', '/'))
        # Or alternatively, add an 'r' prefix to string literals with file paths
        code = code.replace(f"'{filePath}'", f"r'{filePath}'")
        code = code.replace(f'"{filePath}"', f'r"{filePath}"')
    
    # Fix Excel format errors
    if "Excel file format cannot be determined" in error:
        code = code.replace("pd.ExcelFile(", "pd.ExcelFile(engine='openpyxl', ")
        code = re.sub(r'pd\.read_excel\(([^,\)]+)', r'pd.read_excel(\1, engine="openpyxl"', code)
    
    system_content = f'''rewrite the python code.
file path is {filePath} and columns are {columnsName}.
*** IMPORTANT: When using Windows file paths, use one of these approaches:
1. Use raw strings with 'r' prefix: r'C:\\path\\to\\file.xlsx'
2. Use forward slashes: 'C:/path/to/file.xlsx'
3. Use double backslashes: 'C:\\\\path\\\\to\\\\file.xlsx'

IMPORTANT FOR EXCEL FILES:
1. ALWAYS use engine='openpyxl' parameter with pd.read_excel() and pd.ExcelFile()
2. Example: pd.read_excel('file.xlsx', engine='openpyxl', sheet_name='Sheet1')
***'''

    # Add multi-sheet information if available
    if file_info and file_info.get('sheets_info'):
        sheets_info = file_info['sheets_info']
        sheets_text = "This Excel file contains multiple sheets:\n"
        for idx, sheet in enumerate(sheets_info):
            sheets_text += f"Sheet {idx+1}: '{sheet['name']}' - {sheet['shape'][0]} rows, {sheet['shape'][1]} columns, columns: {sheet['columns']}\n"
        system_content += f"\n{sheets_text}\n"
    
    system_content += f'''
code:
{code}
error:
{error}
*update code according to error and code given also use user query if require.
*code can have more errors so fix that as well.
*always load the data and import necessary library.
*for plotting you can use matplotlib and seaborn library only.
*response json has two key 'response' and 'code'.
*while generating code please keep proper indentation. because 'code' key value will be directly executing
*ALWAYS use forward slashes (/) in file paths to avoid unicode escape errors, or use raw strings (r'path\\to\\file')

For Excel files with multiple sheets, ensure you:
1. ALWAYS specify engine='openpyxl' parameter
2. Specify sheet_name parameter correctly
3. Check if the sheet exists before accessing it
4. Consider using pd.ExcelFile for multiple sheet operations'''

    messages = [{"role": "system", "content": system_content}]
    messages.extend(user_query)
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        temperature=0,
        messages=messages
    )
    print(response.usage.total_tokens)
    return response.choices[0].message.content


def pythonCodeTextSummarization(user_query, outputText, columnsName, filePath, file_info=None):
    system_content = f'''we have generated python code for {filePath} file and columns are {columnsName}.'''
    
    # Add multi-sheet information if available
    if file_info and file_info.get('sheets_info'):
        sheets_info = file_info['sheets_info']
        sheets_text = "This Excel file contains multiple sheets:\n"
        for sheet in sheets_info:
            sheets_text += f"Sheet '{sheet['name']}' with columns: {sheet['columns']}\n"
        system_content += f"\n{sheets_text}\n"
    
    system_content += f'''
    * code is generated according to user question and the output of that code is:
    "{str(outputText[:1500])}" 
    
    Provide a CONCISE response that:
    1. Directly answers the user's question in 1-2 sentences
    2. Mentions only the most important numerical results or findings
    3. Uses bullet points for any additional details
    4. Keeps the response under 50 words unless user specifically requested detailed analysis
    
    DO NOT:
    - Include code or technical implementation details
    - Explain methodologies unless explicitly asked
    - Mention the file path or column names
    '''
    
    messages = [{"role": "system", "content": system_content}]
    messages.extend(user_query)
    print(messages)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    print(response.usage.total_tokens)
    return response.choices[0].message.content

def pythonPlotSummarization(user_query, image_path):
    base64_image = local_image_to_data_url("public/" + image_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "user", "content": [
                {
                    "type": "text",
                    "text": '''The given image is generated in previous function using python based on user question.
                * Your job is to provide a COMPREHENSIVE explanation of this visualization:
                1. Describe what the chart shows and how it relates to the user's question
                2. Highlight key patterns, trends, or outliers visible in the chart
                3. Provide business or practical implications of these findings
                4. Be concise but thorough in your explanation
                
                * Don't generate any codes
                * This response will be directly shown to the user
                '''
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"{base64_image}"
                    }
                }
            ]},
            {"role": "user", "content": user_query}
        ],
    )
    return response.choices[0].message.content

def dewasCsvReader(user_query, filePath):
    global conversationalMessage
    conversationalMessage = conversationalMessage[-6:]
    conversationalMessage.append({"role": "user", "content": user_query})
    
    # Initialize file_info dictionary to store metadata
    file_info = {'file_type': None, 'sheets_info': None}
    
    # Handle different file types
    if filePath.lower().endswith('.csv'):
        file_info['file_type'] = 'csv'
        data = pd.read_csv(filePath)
        columns_info = data.dtypes
        columnsName = ', '.join(f"'{col}' ({dtype})" for col, dtype in columns_info.items())
        fileshape = str(data.shape)
    elif filePath.lower().endswith(('.xlsx', '.xls')):
        file_info['file_type'] = 'excel'
        # Get information about all sheets
        file_info['sheets_info'] = get_excel_sheets_info(filePath)
        
        # Get first sheet info for initial context
        first_sheet = file_info['sheets_info'][0]['name'] if file_info['sheets_info'] else None
        if first_sheet:
            data = pd.read_excel(filePath, sheet_name=first_sheet)
            columns_info = data.dtypes
            columnsName = ', '.join(f"'{col}' ({dtype})" for col, dtype in columns_info.items())
            fileshape = str(data.shape)
            columnsName += f" (from sheet '{first_sheet}')"
        else:
            return {'response': f"Unable to read Excel file sheets", 'plot': None}
    else:
        return {'response': f"File format not supported. please upload other file", 'plot': None}
    
    # Extract data insights including common tags
    file_info['data_insights'] = get_data_insights(filePath, file_info)
    
    # Check if query is complex and should be split
    query_analysis = split_complex_query(user_query, filePath, file_info)
    
    # Inside the dewasCsvReader function, replace the if block for complex queries with this:
    if query_analysis.get('is_complex', False):
        print(f"Complex query detected. Split into {len(query_analysis['sub_queries'])} sub-queries")
        
        # Process each sub-query
        analyses = []
        plots = []
        combined_response = ""
        
        for i, sub_query in enumerate(query_analysis['sub_queries']):
            print(f"Processing sub-query {i+1}: {sub_query}")
            
            # Create a temporary conversation context for this sub-query
            temp_conversation = conversationalMessage[:-1] + [{"role": "user", "content": sub_query}]
            
            # Process the sub-query
            sub_result = process_single_query(sub_query, temp_conversation, columnsName, filePath, fileshape, file_info)
            
            # Extract response text
            response_text = sub_result['response']
            if isinstance(response_text, dict):
                # If response is a dictionary, try to get a meaningful text from it
                if 'text' in response_text:
                    response_text = response_text['text']
                elif 'content' in response_text:
                    response_text = response_text['content']
                else:
                    # Convert dict to string as a fallback
                    response_text = str(response_text)
            
            # Add heading for this analysis
            heading = f"### Analysis {i+1}: {query_analysis['sub_queries'][i]}\n\n"
            analysis_with_heading = heading + response_text
            
            # Add to combined response
            combined_response += analysis_with_heading + "\n\n"
            
            # Store analysis result with exact format requested
            analyses.append({
                'analysis': analysis_with_heading,
                'plot': sub_result['plot']
            })
            
            # Collect plot paths
            if sub_result['plot']:
                plots.append(sub_result['plot'])
        
        # Add the combined response to the conversation history
        conversationalMessage.append({"role": "assistant", "content": combined_response})
        
        # Return in the exact format requested
        return {'response': combined_response, 'plot': plots if plots else None, 'analyses': analyses}

    else:
        # Process as a single query
        result = process_single_query(user_query, conversationalMessage, columnsName, filePath, fileshape, file_info)
        
        # Ensure response is a string
        if isinstance(result['response'], dict):
            if 'text' in result['response']:
                result['response'] = result['response']['text']
            elif 'content' in result['response']:
                result['response'] = result['response']['content']
            else:
                result['response'] = str(result['response'])
                
        conversationalMessage.append({"role": "assistant", "content": result['response']})
        return result

def process_single_query(user_query, conversation_context, columnsName, filePath, fileshape, file_info):
    # Generate code
    res = pythonCodeGenerator(
        user_query=conversation_context, 
        columnsName=columnsName, 
        filePath=filePath, 
        fileshape=fileshape,
        file_info=file_info
    )
    
    response = json.loads(res)
    
    if response["code"] is not None:
        outputText, error, plotPath = codeExecutor(str(response["code"]))
        
        if error is not None:
            # Try to fix the code
            res = pythonErrorResolver(
                error=error, 
                user_query=conversation_context, 
                code=str(response["code"]), 
                columnsName=columnsName, 
                filePath=filePath,
                file_info=file_info
            )
            
            response = json.loads(res)
            if response["code"] is not None:
                outputText, error, plotPath = codeExecutor(str(response["code"]))
                
                if error is not None:
                    textResponse = "Sorry, I'm unable to process this request. There seems to be an issue with accessing the data."
                    plot = None
                elif plotPath is None:
                    # Summarize text output
                    textResponse = pythonCodeTextSummarization(
                        user_query=conversation_context, 
                        outputText=outputText, 
                        columnsName=columnsName, 
                        filePath=filePath,
                        file_info=file_info
                    )
                    plot = None
                else:
                    # Summarize plot
                    textResponse = pythonPlotSummarization(user_query=user_query, image_path=plotPath)
                    plot = plotPath
                
            elif response["response"] is not None:
                textResponse = response["response"]
                plot = None
            else:
                textResponse = "Sorry, I'm unable to process this request."
                plot = None
                
        elif plotPath is None:
            # Summarize text output
            textResponse = pythonCodeTextSummarization(
                user_query=conversation_context, 
                outputText=outputText, 
                columnsName=columnsName, 
                filePath=filePath,
                file_info=file_info
            )
            plot = None
        else:
            # Summarize plot
            textResponse = pythonPlotSummarization(user_query=user_query, image_path=plotPath)
            plot = plotPath
    elif response["response"] is not None:
        textResponse = response["response"]
        plot = None
    else:
        textResponse = "Sorry, I'm unable to process this request."
        plot = None
        
    print({'response': textResponse, 'plot': plot})
    return {'response': textResponse, 'plot': plot}